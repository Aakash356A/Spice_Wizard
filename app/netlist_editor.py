"""
Netlist parsing and editing logic.
Extracted from spice_runner.py for reusability.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import re

SUFFIXES = {
    "T": 1e12,
    "G": 1e9,
    "MEG": 1e6,
    "K": 1e3,
    "M": 1e-3,
    "U": 1e-6,
    "N": 1e-9,
    "P": 1e-12,
    "F": 1e-15,
}

ANALYSIS_KINDS = (".ac", ".tran", ".dc")

@dataclass
class Element:
    name: str
    etype: str
    nodes: List[str]
    value_token: Optional[str]
    line_idx: int
    token_idx: Optional[int]

@dataclass
class ParamDef:
    key: str
    value_token: str
    line_idx: int
    token_span: Tuple[int, int]

@dataclass
class AnalysisCard:
    kind: str
    args: str
    line_idx: int

@dataclass
class NetlistModel:
    lines: List[str]
    elements: Dict[str, Element]
    params: Dict[str, ParamDef]
    analyses: List[AnalysisCard]

def is_comment(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return s.startswith('*') or s.startswith(';') or s.lower().startswith('//')

def split_tokens(line: str) -> List[str]:
    return re.findall(r'\S+', line)

def parse_netlist(text: str) -> NetlistModel:
    lines = text.splitlines(keepends=False)
    elements: Dict[str, Element] = {}
    params: Dict[str, ParamDef] = {}
    analyses: List[AnalysisCard] = []

    for idx, raw in enumerate(lines):
        line = raw.rstrip('\n')
        if not line.strip():
            continue
        if is_comment(line):
            continue

        if line.lstrip().lower().startswith('.param'):
            body = line.lstrip()[len('.param'):].strip()
            for m in re.finditer(r'([A-Za-z_]\w*)\s*=\s*([^\s]+)', body):
                key = m.group(1)
                val = m.group(2)
                start = line.find(m.group(0))
                end = start + len(m.group(0))
                params[key.upper()] = ParamDef(key=key, value_token=val, line_idx=idx, token_span=(start, end))
            continue

        ls = line.lstrip()
        for kind in ANALYSIS_KINDS:
            if ls.lower().startswith(kind):
                args = ls[len(kind):].strip()
                analyses.append(AnalysisCard(kind=kind, args=args, line_idx=idx))
                break
        else:
            tokens = split_tokens(line)
            if not tokens:
                continue
            name = tokens[0]
            if not name:
                continue
            etype = name[0].upper()
            if etype in ('R','C','L'):
                if len(tokens) >= 4:
                    nodes = [tokens[1], tokens[2]]
                    value_tok = tokens[3]
                    elements[name.upper()] = Element(
                        name=name, etype=etype, nodes=nodes,
                        value_token=value_tok, line_idx=idx, token_idx=3
                    )
    return NetlistModel(lines=lines, elements=elements, params=params, analyses=analyses)

def set_element_value(model: NetlistModel, name: str, new_token: str) -> bool:
    el = model.elements.get(name.upper())
    if not el:
        return False
    line = model.lines[el.line_idx]
    toks = split_tokens(line)
    if el.token_idx is None or el.token_idx >= len(toks):
        return False
    old_tok = toks[el.token_idx]
    pattern = re.escape(old_tok)
    new_line = re.sub(pattern, new_token, line, count=1)
    model.lines[el.line_idx] = new_line
    el.value_token = new_token
    return True

def set_param_value(model: NetlistModel, key: str, new_token: str) -> bool:
    p = model.params.get(key.upper())
    if not p:
        return False
    line = model.lines[p.line_idx]
    if '.param' not in line.lower():
        return False
    body = line.lstrip()[len('.param'):]
    pairs = list(re.finditer(r'([A-Za-z_]\w*)\s*=\s*([^\s]+)', body))
    new_pairs = []
    for m in pairs:
        k = m.group(1)
        v = m.group(2)
        if k.upper() == key.upper():
            v = new_token
        new_pairs.append(f"{k}={v}")
    indent_len = len(line) - len(line.lstrip())
    indent = line[:indent_len]
    new_line = indent + ".param " + "  ".join(new_pairs)
    model.lines[p.line_idx] = new_line
    refreshed = parse_netlist("\n".join(model.lines))
    model.params = refreshed.params
    return True

def set_analysis(model: NetlistModel, kind: str, args: str) -> None:
    """
    Updates the analysis card in the model.
    It finds the last active analysis command, replaces it, and comments out any others.
    If no active analysis is found, it appends a new one.
    """
    kind_low = kind.lower()
    if kind_low not in ANALYSIS_KINDS:
        raise ValueError(f"Unsupported analysis kind: {kind}")

    new_line = f"{kind_low} {args}".rstrip()
    
    # Find all non-commented analysis cards
    active_analyses = [ac for ac in model.analyses if not is_comment(model.lines[ac.line_idx])]

    if active_analyses:
        # Use the last active analysis as the one to replace
        target_ac = active_analyses.pop()
        model.lines[target_ac.line_idx] = new_line
        
        # Comment out any other active analyses
        for other_ac in active_analyses:
            line = model.lines[other_ac.line_idx]
            model.lines[other_ac.line_idx] = f"* Edited by GUI: {line}"
    else:
        # If no active analysis was found, append a new one
        model.lines.append(new_line)

    # After modification, re-parse to update the model's analysis list accurately
    refreshed_model = parse_netlist("\n".join(model.lines))
    model.analyses = refreshed_model.analyses

def save_netlist(model: NetlistModel, output_path: Path) -> None:
    """Save the edited netlist to a file."""
    from datetime import datetime
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"* Edited by SPICE GUI on {datetime.now().isoformat(timespec='seconds')}\n")
        for ln in model.lines:
            f.write(ln.rstrip('\n') + '\n')
