import json
import re
import os
import requests
import difflib
from typing import Dict, List, Optional
from dotenv import load_dotenv
from ft_mac import LocalGemmaGen

load_dotenv()  # loads OPENROUTER_API_KEY (and friends) from a local .env file, if present

# Configuration from environment or defaults
OPENROUTER_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_OPENROUTER_MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENROUTER_MODEL", "openai/gpt-5")
OPENROUTER_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")


class UnifiedAgent:
    def __init__(self):
        # 1. Initialize the Specialist (Local Fine-Tuned Model)
        self.specialist = LocalGemmaGen()
        
        # 2. State memory
        self.last_netlist = None
        self.last_ic = None
        self.conversation_history = []  # Already exists, but will now be used
        
        # 3. Load ADI IC Database for RAG and Validation
        self.ic_database = self._load_ic_database()

        # Ensure API Key is available
        if not OPENROUTER_API_KEY:
            print("WARNING: OPENROUTER_API_KEY not set. General LLM features will fail.")

    def _load_ic_database(self) -> List[str]:
        """Loads local text file containing the authorized IC list."""
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "available_ics.txt")
        ics = []
        if os.path.exists(db_path):
            try:
                with open(db_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Skip headers or empty lines
                        if line and not line.startswith("=") and "Available ICs" not in line:
                             ics.append(line)
                print(f"Loaded {len(ics)} ICs from validation list.")
                return ics
            except Exception as e:
                print(f"Error loading IC database: {e}")
        return []

    def _find_relevant_ics(self, query: str) -> str:
        """
        Find relevant ICs from the local database based on user query.
        Uses fuzzy matching on potential part numbers.
        """
        if not self.ic_database:
            return ""

        # Extract things that look like part numbers (e.g. "LT1028", "AD8221", "MAX123")
        potential_parts = re.findall(r'\b[A-Za-z]{2,}\d+[A-Za-z0-9\-]*\b', query.upper())
        
        matches = set()
        
        # Check against DB
        for part in potential_parts:
             # Find close matches (handling typos like "LT1028" vs "LT1028A")
             close = difflib.get_close_matches(part, self.ic_database, n=10, cutoff=0.6)
             matches.update(close)
             
             # Also check starts_with for partials like "LT30"
             for ic in self.ic_database:
                 if ic.startswith(part):
                     matches.add(ic)
                     if len(matches) > 20: break

        # Limit context size
        final_list = sorted(list(matches))[:30]
        
        if not final_list:
            return ""

        result = "\nCONTEXT - AVAILABLE ICs MATCHING USER INPUT:\n"
        result += ", ".join(final_list)
        result += "\n(Prefer selecting from this list if the user is asking for these specific parts)\n"
        
        return result

    def _validate_ic_name(self, ic_name: str) -> str:
        """Validate and auto-correct IC name against database."""
        if not ic_name or not self.ic_database:
            return ic_name
            
        # Exact match check
        if ic_name.upper() in [ic.upper() for ic in self.ic_database]:
            # Return the correctly cased version
            for ic in self.ic_database:
                if ic.upper() == ic_name.upper():
                    return ic
            return ic_name
            
        # Fuzzy fix: try to find closest match in DB
        matches = difflib.get_close_matches(ic_name.upper(), [ic.upper() for ic in self.ic_database], n=1, cutoff=0.7)
        if matches:
            # Find original casing
            for ic in self.ic_database:
                if ic.upper() == matches[0]:
                    print(f"AGENT: Auto-correcting IC name '{ic_name}' -> '{ic}'")
                    return ic
        
        return ic_name

    def _call_general_llm(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str | Dict:
        """
        Calls the smart General LLM (GPT-4o, Claude, etc) via OpenRouter.
        Now includes conversation history for context.
        """
        endpoint = f"{OPENROUTER_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/circuit-llm",
            "X-Title": "ADI Agent",
        }
                
        # LOGGING INPUT
        print(f"\n{'='*20} [UnifiedAgent] REQUEST TO OPENROUTER {'='*20}")
        print(f"MODEL: {DEFAULT_OPENROUTER_MODEL}")
        print(f"--- SYSTEM PROMPT ---\n{system_prompt}")
        print(f"--- USER PROMPT ---\n{user_prompt}")
        print(f"{'='*60}\n")
        
        # Build messages: system + history + current user
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)  # Add full history
        messages.append({"role": "user", "content": user_prompt})
        
        payload = {
            "model": DEFAULT_OPENROUTER_MODEL,
            "messages": messages,  # Now includes history
            "temperature": 0.2 if json_mode else 0.7,
            "max_tokens": 15000,
            "top_p": 0.9,
        }
        
        # Add reasoning parameter if using GPT-5/o1 models that support it
        if "gpt-5" in DEFAULT_OPENROUTER_MODEL.lower() or "o1" in DEFAULT_OPENROUTER_MODEL.lower():
            payload["reasoning"] = {"effort": "high"}

        try:
            print(f"DEBUG: Calling OpenRouter with model {DEFAULT_OPENROUTER_MODEL}...")
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"]
                
                # LOGGING OUTPUT
                print(f"\n{'='*20} [UnifiedAgent] RESPONSE FROM OPENROUTER {'='*20}")
                print(f"{content}")
                print(f"{'='*60}\n")
                
                if json_mode:
                    return self._extract_json(content)
                return content
            return {"error": "Empty response from OpenRouter"}
            
        except Exception as e:
            print(f"ERROR: General LLM call failed: {e}")
            return {"error": str(e)}

    def _extract_json(self, content: str) -> Dict:
        """Helper to safely parse JSON from LLM output."""
        if isinstance(content, dict): return content
        
        txt = content.strip()
        # Clean markdown code blocks
        if "```json" in txt:
            txt = txt.split("```json")[1].split("```")[0].strip()
        elif "```" in txt:
            txt = txt.split("```")[1].split("```")[0].strip()
            
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            # Try to find brace boundaries
            start = txt.find("{")
            end = txt.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(txt[start:end+1])
                except:
                    pass
            return {"error": f"Failed to parse JSON: {content[:100]}..."}

    def analyze_intent(self, user_query: str) -> Dict:
        """
        Ask the General LLM to classify the user's request.
        Now supports multi-IC subsystem detection.
        """
        # 1. RETRIEVE RELEVANT CONTEXT (Names only)
        suggested_ics_text = self._find_relevant_ics(user_query)
        
        system_prompt = f"""
You are an expert SPICE Netlist Agent Planner specializing in Analog Devices (ADI) products. 
Analyze the user's request and determine the best course of action.

{suggested_ics_text}

**CRITICAL RULES:**
1. If the user asks for a SUBSYSTEM or CIRCUIT requiring MULTIPLE ICs (e.g., "voltage reference with buffer", "DAC with filter and amplifier"), you MUST identify ALL required ICs.
2. Select appropriate ADI parts for each function. Prefer parts from the context list if available.
3. For multi-IC requests, list each IC separately in the "ic_names" array.

Return a **JSON object** (no markdown) with these fields:

1. "action": One of:
   - "generate_single": User wants a standard circuit with ONE IC.
   - "generate_subsystem": User wants a SUBSYSTEM requiring MULTIPLE ICs (2 or more).
   - "modify": User wants to modify an existing netlist.
   - "combine": User wants to merge existing circuits.
   - "chat": General questions not requiring netlist action.

2. "ic_names": An ARRAY of IC model numbers required. Examples:
   - Single IC: ["AD8221"]
   - Subsystem: ["ADR4525", "ADA4077-1"]  (reference + buffer)
   - Subsystem: ["AD5691R", "ADA4891-1"]  (DAC + output buffer)
   
3. "subsystem_description": For "generate_subsystem" action, describe how the ICs connect:
   - Example: "ADR4525 provides 2.5V reference output -> ADA4077-1 configured as unity-gain buffer"
   
4. "modification_instruction": If user provided specific values (e.g. "gain of 10", "5V output"), extract them here.

5. "reply": A short sentence explaining your plan (e.g., "I will generate a precision reference subsystem using ADR4525 and ADA4077-1 as buffer.").
"""

        # Call Real LLM
        plan = self._call_general_llm(system_prompt, user_query, json_mode=True)
        
        # Fallback if LLM fails completely
        if "error" in plan:
            print(f"Plan Error: {plan['error']}")
            return {
                "action": "chat", 
                "reply": "I'm having trouble connecting to my brain. Please check your API key."
            }
        
        # Validate all IC names
        if "ic_names" in plan and isinstance(plan["ic_names"], list):
            plan["ic_names"] = [self._validate_ic_name(ic) for ic in plan["ic_names"]]
        
        # Backwards compatibility: if old "ic_name" field used
        if "ic_name" in plan and not plan.get("ic_names"):
            plan["ic_names"] = [self._validate_ic_name(plan["ic_name"])]
            
        return plan

    def _generate_single_netlist(self, ic_name: str) -> str:
        """Generate netlist for a single IC using the specialist model."""
        print(f"\n{'#'*20} [AGENT] Requesting netlist for: {ic_name} {'#'*20}")
        raw_netlist = self.specialist.generate_netlist(ic_name)
        return raw_netlist

    def _combine_netlists_for_subsystem(self, netlists: List[Dict[str, str]], subsystem_description: str, user_query: str) -> str:
        """
        Use the General LLM to intelligently combine multiple IC netlists into one subsystem.
        """
        combined_text = ""
        for item in netlists:
            combined_text += f"\n{'='*40}\n"
            combined_text += f"IC: {item['ic_name']}\n"
            combined_text += f"{'='*40}\n"
            combined_text += item['netlist']
            combined_text += "\n"

        merge_prompt = f"""
You are an expert SPICE netlist engineer. Combine the following individual IC netlists into ONE unified subsystem netlist.

**ORIGINAL USER REQUEST:**
{user_query}

**SUBSYSTEM ARCHITECTURE:**
{subsystem_description}

**INDIVIDUAL IC NETLISTS:**
{combined_text}

**YOUR TASK:**
1. Merge all netlists into ONE valid SPICE netlist.
2. Connect the ICs according to the subsystem architecture:
   - Identify output nodes of upstream ICs
   - Connect them to input nodes of downstream ICs
   - Rename nodes to avoid conflicts (e.g., if both have "out", rename to "vref_out", "buffer_out")
3. Keep all .lib and .model references
4. Use a single .tran or .ac analysis (choose appropriate one)
5. Add a meaningful title comment describing the subsystem
6. Ensure proper ground connections (node 0 or GND)

WARNING: Do NOT invent new components or ICs. Only use the provided netlists.
**OUTPUT FORMAT:**
Return ONLY the merged SPICE netlist. No markdown, no explanations, no code blocks.
Start directly with the netlist title comment (e.g., "* Precision Reference with Buffer Subsystem")
"""

        merge_system = "You are a SPICE netlist expert. Output ONLY valid SPICE netlist text with no markdown or explanations."
        
        merged_resp = self._call_general_llm(merge_system, merge_prompt, json_mode=False)
        
        if isinstance(merged_resp, dict) and "error" in merged_resp:
            raise Exception(merged_resp["error"])
        
        # Cleanup potential markdown wrapper
        merged_netlist = str(merged_resp).strip()
        if "```" in merged_netlist:
            merged_netlist = merged_netlist.replace("```spice", "").replace("```net", "").replace("```", "").strip()
        
        return merged_netlist

    def process_request(self, user_query: str) -> Dict:
        """
        Main entry point for the GUI.
        Now appends to conversation history for context in future calls.
        """
        # Append user message to history
        self.conversation_history.append({"role": "user", "content": user_query})
        
        # 1. Plan the Task
        plan = self.analyze_intent(user_query)
        action = plan.get("action")
        ic_names = plan.get("ic_names", [])
        
        result_response = {
            "mode": action,
            "IC": ", ".join(ic_names) if ic_names else None,
            "Summary": plan.get("reply"),
            "netlist": None,
            "error": None
        }
        
        try:
            # ============== SINGLE IC GENERATION ==============
            if action == "generate_single":
                if not ic_names:
                    result_response["Summary"] = "I couldn't identify which IC you want."
                    return result_response
                
                ic_name = ic_names[0]
                print(f"AGENT: Routing to Specialist for single IC: {ic_name}")
                raw_netlist = self._generate_single_netlist(ic_name)
                
                # NEW: Check if there are modification instructions to apply
                mod_instruction = plan.get("modification_instruction")
                if mod_instruction and mod_instruction.strip():
                    print(f"AGENT: Applying modification instructions to generated netlist...")
                    mod_prompt = f"""
Here is a base SPICE netlist for {ic_name}:
{raw_netlist}

Apply these specific modifications:
{mod_instruction}

Original user request: {user_query}

Return ONLY the modified netlist text. No markdown, no explanations, no code blocks.
Start directly with netlist content. Ensure all requested components (capacitors, resistors, etc.) are added.
"""
                    modified_resp = self._call_general_llm(
                        "You are a SPICE expert. Output ONLY raw netlist text with the requested modifications applied.", 
                        mod_prompt, 
                        json_mode=False
                    )
                    
                    if isinstance(modified_resp, dict) and "error" in modified_resp:
                        print(f"AGENT: Modification failed, using base netlist: {modified_resp['error']}")
                        # Fall back to raw netlist
                    else:
                        # Cleanup potential markdown wrapper
                        raw_netlist = str(modified_resp).strip()
                        if "```" in raw_netlist:
                            raw_netlist = raw_netlist.replace("```spice", "").replace("```net", "").replace("```", "").strip()
                
                self.last_netlist = raw_netlist
                self.last_ic = ic_name
                result_response["netlist"] = raw_netlist

            # ============== MULTI-IC SUBSYSTEM GENERATION ==============
            elif action == "generate_subsystem":
                if len(ic_names) < 2:
                    result_response["Summary"] = "Subsystem requires at least 2 ICs, but I only found one."
                    # Fallback to single generation
                    if ic_names:
                        raw_netlist = self._generate_single_netlist(ic_names[0])
                        self.last_netlist = raw_netlist
                        self.last_ic = ic_names[0]
                        result_response["netlist"] = raw_netlist
                    return result_response
                
                print(f"AGENT: === SUBSYSTEM GENERATION MODE ===")
                print(f"AGENT: ICs required: {ic_names}")
                
                # Step 1: Generate netlist for EACH IC sequentially
                individual_netlists = []
                for ic_name in ic_names:
                    print(f"AGENT: [Step {len(individual_netlists)+1}/{len(ic_names)}] Generating netlist for {ic_name}...")
                    netlist = self._generate_single_netlist(ic_name)
                    individual_netlists.append({
                        "ic_name": ic_name,
                        "netlist": netlist
                    })
                    print(f"AGENT: ✓ Got netlist for {ic_name} ({len(netlist)} chars)")
                
                # Step 2: Combine netlists using General LLM
                subsystem_desc = plan.get("subsystem_description", "Connect ICs in series as described by user.")
                print(f"AGENT: Combining {len(individual_netlists)} netlists into subsystem...")
                
                combined_netlist = self._combine_netlists_for_subsystem(
                    individual_netlists, 
                    subsystem_desc,
                    user_query
                )
                
                self.last_netlist = combined_netlist
                self.last_ic = "+".join(ic_names)
                result_response["netlist"] = combined_netlist
                result_response["Summary"] = f"Generated subsystem with {len(ic_names)} ICs: {', '.join(ic_names)}"

            # ============== MODIFICATION ==============
            elif action == "modify":
                ic_name = ic_names[0] if ic_names else None
                
                # Fallback to last IC if context implies modification of current
                if not ic_name and self.last_ic:
                    ic_name = self.last_ic

                if ic_name and "+" not in (self.last_ic or ""):
                    # Generate fresh if not modifying a subsystem
                    print(f"AGENT: Step 1 - Fetching base {ic_name}")
                    base_netlist = self._generate_single_netlist(ic_name)
                elif self.last_netlist:
                    print(f"AGENT: Using previous netlist context.")
                    base_netlist = self.last_netlist
                    ic_name = self.last_ic
                else:
                    result_response["Summary"] = "I don't have a netlist to modify. Please request an IC first."
                    return result_response
                
                # Pass base + instruction to General LLM
                mod_prompt = f"""
Here is a SPICE netlist:
{base_netlist}

User Request: {plan.get('modification_instruction', user_query)}

Return ONLY the modified netlist text. No markdown, no explanations, no code blocks. 
Start directly with netlist content.
"""
                print(f"AGENT: Step 2 - asking General LLM to modify...")
                
                modified_netlist_resp = self._call_general_llm(
                    "You are a SPICE expert. Output ONLY raw netlist text.", 
                    mod_prompt, 
                    json_mode=False
                )
                
                if isinstance(modified_netlist_resp, dict) and "error" in modified_netlist_resp:
                    raise Exception(modified_netlist_resp["error"])
                
                # Cleanup potential markdown wrapper
                modified_netlist = str(modified_netlist_resp).strip()
                if "```" in modified_netlist:
                    modified_netlist = modified_netlist.replace("```spice", "").replace("```net", "").replace("```", "").strip()
                
                self.last_netlist = modified_netlist
                self.last_ic = ic_name
                result_response["netlist"] = modified_netlist

            elif action == "chat":
                # Just talk - the 'reply' comes from the planner
                pass 

            # After processing, append assistant response to history
            assistant_content = f"Action: {action}, ICs: {result_response['IC'] or 'N/A'}, Summary: {result_response['Summary'] or 'N/A'}"
            if result_response.get("netlist"):
                assistant_content += f"\nNetlist: {result_response['netlist'][:200]}..."  # Truncate for brevity
            self.conversation_history.append({"role": "assistant", "content": assistant_content})
            
        except Exception as e:
            result_response["error"] = str(e)
            # Append error to history
            self.conversation_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
            import traceback
            traceback.print_exc()
            
        return result_response

    def merge_netlists(self, netlist_list: list, instructions: str):
        """
        Specific helper for the combine tab (manual merging).
        """
        combined_text = ""
        for n in netlist_list:
            combined_text += f"\n=== NETLIST: {n['label']} ===\n{n['text']}\n"
            
        prompt = f"""
Merge the following SPICE netlists together into one functional circuit.

{combined_text}

Instructions: {instructions}

Ensure unique node names where necessary to prevent short circuits, but connect nodes explicitly mentioned in instructions.
Return ONLY the valid joined netlist. No explanation.
"""
        
        merged_resp = self._call_general_llm(
            "You are a SPICE merging expert. Output ONLY raw netlist text.", 
            prompt, 
            json_mode=False
        )
        
        if isinstance(merged_resp, dict) and "error" in merged_resp:
            return {"error": merged_resp["error"]}
             
        merged = str(merged_resp).strip()
        if "```" in merged:
            merged = merged.replace("```spice", "").replace("```", "").strip()
        
        return {"mode": "merge", "netlist": merged, "Summary": "Merging complete."}

    def show_history(self):
        return self.conversation_history

    def reset_conversation(self, keep_system=True):
        # Already clears history, but now it's populated
        self.conversation_history = []
        self.last_netlist = None
        self.last_ic = None