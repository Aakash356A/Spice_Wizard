# train_ltspice_sft.py
# Finetune (Unsloth + TRL SFT) for STRICT netlist output with solid splits, eval, metrics & regex constraints.

import os, re, json, argparse, random, hashlib
from typing import Dict, List, Tuple
import numpy as np

import torch
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer

# Unsloth
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("ACCELERATE_DISABLE_TORCHAO", "1")
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

# -------------------------
# Regex helpers (validation & constrained decoding)
# -------------------------

# Allowed LTspice line prefixes (extend as needed)
ALLOWED_PREFIX = re.compile(r"^(R|C|L|D|V|I|X|Q|M|K|E|F|G|H|\\.\\w+|\\*).*$")
# Mandatory directives you want to enforce (tweak to your style)
MANDATORY_DIRECTIVES = [r"^\\.end$"]
MANDATORY_COMPILED = [re.compile(pat) for pat in MANDATORY_DIRECTIVES]

# A quick "is this probably a pure netlist?" check (no extra prose, lines look plausible)
def is_parseable_netlist(s: str) -> bool:
    lines = [ln.rstrip() for ln in s.strip().splitlines() if ln.strip() != ""]
    if not lines:
        return False
    # All lines either comments (*...), components, or dot-directives
    for ln in lines:
        if not ALLOWED_PREFIX.match(ln):
            return False
    # Must contain .end and any other directives you deem critical
    for pat in MANDATORY_COMPILED:
        if not any(pat.match(ln) for ln in lines):
            return False
    return True

# Regex-constrained generation via simple rejection sampling (local, no TGI server required):
def constrained_generate(generate_fn, prompt_ids, tokenizer, max_new_tokens=256, num_attempts=8) -> str:
    """generate_fn: lambda **kwargs -> token ids (like model.generate)
       prompt_ids: tokenized prompt (input_ids tensor)
       Returns first generation that passes is_parseable_netlist(), else last candidate."""
    for i in range(num_attempts):
        out = generate_fn(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7 if i == 0 else 0.9,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0][prompt_ids.shape[-1]:], skip_special_tokens=True)
        if is_parseable_netlist(text):
            return text
    return text  # fallback last try

# -------------------------
# Dedup & canonicalization to avoid leakage
# -------------------------

def canonicalize_netlist_text(s: str) -> str:
    # Strip trailing spaces, normalize multiple spaces, drop backanno lines if desired
    out_lines = []
    for ln in s.splitlines():
        # Optional: strip comments IF they cause near-duplicate leakage.
        # Keep '*' comment lines if you consider them part of the ground truth.
        # Example: ignore .backanno and empty lines
        if ln.strip().lower().startswith(".backanno"):
            continue
        # normalize whitespace
        ln = re.sub(r"\s+", " ", ln.strip())
        if ln:
            out_lines.append(ln)
    return "\n".join(out_lines).strip()

def dedup_by_output(ds: Dataset, output_key="output") -> Dataset:
    seen = set()
    kept_rows = []
    for row in ds:
        can = canonicalize_netlist_text(row[output_key])
        h = hashlib.sha1(can.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            kept_rows.append(row)
    return Dataset.from_list(kept_rows)

# -------------------------
# Formatting to a single "text" column using Unsloth chat templates
# -------------------------

def build_formatter(tokenizer, response_only: bool, add_generation_prompt: bool, response_part: str):
    def _format_row(row):
        messages = [
            {"role": "system",    "content": row.get("system", "You are an expert at generating LTspice netlists. Return ONLY the netlist with no extra text.")},
            {"role": "user",      "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]},
        ]
        txt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return {"text": txt + tokenizer.eos_token}
    return _format_row

# -------------------------
# Metrics: Exact-Match, Parseable %, Required-directives %
# -------------------------

def compute_task_metrics(pred_texts: List[str], label_texts: List[str]) -> Dict[str, float]:
    assert len(pred_texts) == len(label_texts)
    n = len(pred_texts)
    em = sum(int(p.strip() == g.strip()) for p, g in zip(pred_texts, label_texts)) / max(n, 1)
    parse_rate = sum(int(is_parseable_netlist(p)) for p in pred_texts) / max(n, 1)

    def has_all_directives(s: str) -> bool:
        lines = [ln.rstrip() for ln in s.strip().splitlines()]
        return all(any(pat.match(ln) for ln in lines) for pat in MANDATORY_COMPILED)

    req_rate = sum(int(has_all_directives(p)) for p in pred_texts) / max(n, 1)
    return {"exact_match": em, "parseable_rate": parse_rate, "required_directives_rate": req_rate}

# -------------------------
# Main
# -------------------------

def main():
    p = argparse.ArgumentParser()
    # Data & templates
    p.add_argument("--data_jsonl", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./out_sft")
    p.add_argument("--model_name", type=str, default="unsloth/gpt-oss-20b-unsloth-bnb-4bit")
    p.add_argument("--chat_template", type=str, default="gpt-oss", help="e.g., gpt-oss or gemma-3")
    p.add_argument("--instruction_part", type=str, default="<|start|>user<|message|>")
    p.add_argument("--response_part", type=str, default="<|start|>assistant<|channel|>final<|message|>")
    #--instruction_part "<start_of_turn>user\n"
#--response_part "<start_of_turn>model\n"
    # Splits
    p.add_argument("--test_size", type=float, default=0.10)
    p.add_argument("--val_size", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stratify_by", type=str, default=None, help="optional column to stratify (e.g., 'family')")
    p.add_argument("--dedup", action="store_true", help="deduplicate near-duplicates by canonicalized output")
    # Training knobs
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--lr_scheduler", type=str, default="linear", choices=["linear","cosine","cosine_with_restarts","polynomial","constant"])
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--save_steps", type=int, default=600)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--packing", action="store_true")
    # Loss options
    p.add_argument("--label_smoothing", type=float, default=0.0, help="0 = standard CE; >0 enables label smoothing CE")
    # Precision
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    # Early stopping
    p.add_argument("--early_stop_patience", type=int, default=3)
    # Constrained decoding at eval (simple rejection via regex)
    p.add_argument("--eval_with_regex_constraints", action="store_true")
    args = p.parse_args()

    # ----------------- Load dataset & optional dedup -----------------
    raw = load_dataset("json", data_files=args.data_jsonl, split="train")
    if args.dedup:
        raw = dedup_by_output(raw, output_key="output")

    # Optional: if you want to stratify, ensure the column exists; else None
    test_split = raw.train_test_split(test_size=args.test_size, seed=args.seed,
                                      stratify_by_column=args.stratify_by) if args.stratify_by else \
                 raw.train_test_split(test_size=args.test_size, seed=args.seed)
    train_all, test = test_split["train"], test_split["test"]

    val_rel = args.val_size / max(1e-8, (1.0 - args.test_size))  # fraction of train_all
    tv = train_all.train_test_split(test_size=val_rel, seed=args.seed,
                                    stratify_by_column=args.stratify_by) if args.stratify_by else \
         train_all.train_test_split(test_size=val_rel, seed=args.seed)
    train, val = tv["train"], tv["test"]

    # ----------------- Load model -----------------
    device_map = "auto"
    offload_dir = "offload"; os.makedirs(offload_dir, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = args.model_name,
        dtype          = None,  # Let Unsloth pick (BF16 on Ampere+)
        max_seq_length = args.max_seq_length,
        load_in_4bit   = True,
        device_map     = device_map,
        offload_folder = offload_dir,
    )

    # LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        finetune_vision_layers     = False,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0.0,
        target_modules = ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
        use_rslora = None,
        random_state = args.seed,
        loftq_config = None,
    )

    # Chat template
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    # Format
    fmt = build_formatter(tokenizer, response_only=True, add_generation_prompt=False, response_part=args.response_part)
    train = train.map(fmt, remove_columns=train.column_names)
    val   = val.map(fmt,   remove_columns=val.column_names)
    test  = test.map(fmt,  remove_columns=test.column_names)

    # ----------------- TrainingArguments -----------------
    use_bf16 = args.bf16 or (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    use_fp16 = args.fp16 or (torch.cuda.is_available() and not use_bf16)

    targs = TrainingArguments(
        output_dir = args.output_dir,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        warmup_steps = args.warmup_steps,
        num_train_epochs = args.epochs,
        learning_rate = args.lr,
        logging_steps = args.logging_steps,
        save_steps = args.save_steps,
        save_total_limit = 2,
        lr_scheduler_type = args.lr_scheduler,
        weight_decay = args.weight_decay,
        bf16 = use_bf16,
        fp16 = use_fp16,
        gradient_checkpointing = True,
        report_to = "none",
        # Eval & best checkpoint
        eval_strategy = "steps",
        eval_steps = args.eval_steps,
        do_eval = True,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
        # Label smoothing toggles CE variant under the hood
        label_smoothing_factor = args.label_smoothing,
        # For text-gen metrics
        #predict_with_generate = True,
        #generation_max_length = 512,
    )

    # ----------------- Trainer -----------------
    trainer = SFTTrainer(
        model = model,
       
        
        train_dataset = train,
        eval_dataset  = val,
        #dataset_text_field = "text",
        #max_seq_length = args.max_seq_length,
        #packing = args.packing,
        #tokenizer = tokenizer,
        args = targs,
    )

    # Teach trainer to mask loss on prompts & compute loss on responses only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = args.instruction_part,
        response_part    = args.response_part,
    )
    
    # ---- compute_metrics: decode predictions & labels, then task metrics
    def compute_metrics(eval_pred):
        # eval_pred.predictions are logits; we ignore and instead generate.
        # We'll use the eval_dataset's inputs to re-build prompts and call .generate().
        model.eval()
        preds_text, labels_text = [], []

        # reconstruct the original prompts for eval set:
        # if you kept raw eval rows somewhere, loop them; otherwise reuse tokenized inputs.
        # Example (pseudo): build prompts with tokenizer.apply_chat_template(...)
        for row in raw_val_rows:  # keep a copy before mapping/formatting
            messages = [
                {"role":"system","content":row["system"]},
                {"role":"user","content":row["instruction"]},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors="pt").to(model.device)["input_ids"]

            with torch.no_grad():
                gen = model.generate(
                    input_ids=ids,
                    max_new_tokens=512,
                    do_sample=False,        # use greedy/beam for eval to be stable
                    eos_token_id=tokenizer.eos_token_id,
                )
            out = tokenizer.decode(gen[0][ids.shape[-1]:], skip_special_tokens=True)
            preds_text.append(out.strip())
            labels_text.append(row["output"].strip())

        # Your metrics (exact match / parseable / directives)
        return compute_task_metrics(preds_text, labels_text)

    trainer.compute_metrics = compute_metrics
    # def compute_metrics(eval_pred):
    #     # SFTTrainer returns generated token ids in eval_pred.predictions when predict_with_generate=True
    #     preds = eval_pred.predictions
    #     labels = eval_pred.label_ids

    #     # Decode
    #     pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #     label_texts = tokenizer.batch_decode(np.where(labels != -100, labels, tokenizer.pad_token_id), skip_special_tokens=True)

    #     # Optionally post-trim prompt residues if your template leaves any.
    #     def post_trim(txt: str) -> str:
    #         # Heuristic: keep from first non-comment/component line or first '.'
    #         return txt.strip()
    #     pred_texts = [post_trim(t) for t in pred_texts]
    #     label_texts = [post_trim(t) for t in label_texts]

    #     metrics = compute_task_metrics(pred_texts, label_texts)
    #     return metrics

    # trainer.compute_metrics = compute_metrics
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience))

    trainer.train()

    # ----------------- Evaluate on TEST -----------------
    test_metrics = trainer.evaluate(test)
    print("[TEST] metrics:", test_metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ----------------- Optional: regex-constrained sample demo -----------------
    if args.eval_with_regex_constraints:
        ex = {
            "system": "You are an expert at generating LTspice netlists. Return ONLY the netlist with no extra text.",
            "instruction": "Fetch me a LT1073-5",
        }
        messages = [
            {"role": "system", "content": ex["system"]},
            {"role": "user",   "content": ex["instruction"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt, return_tensors="pt").to(model.device)["input_ids"]

        def gen_fn(**kwargs):
            return model.generate(**kwargs)

        out_text = constrained_generate(gen_fn, prompt_ids, tokenizer, max_new_tokens=256, num_attempts=6)
        print("\n[Constrained demo output]\n", out_text)

if __name__ == "__main__":
    main()
