"""
LTspice simulation runner with threading support for GUI.
"""

from pathlib import Path
from typing import Dict, Optional, List
import subprocess
import time
import os
import sys
import shutil
from threading import Thread
import logging
import platform
import re

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
def ltspice_lib_candidates():
    home = Path.home()
    sysname = platform.system().lower()
    c = []

    # Optional override
    env = os.environ.get("LTSPICE_LIB_PATH")
    if env:
        c.append(Path(env).expanduser())

    if "darwin" in sysname or "mac" in sysname:
        c += [
            home / "Library" / "Application Support" / "LTspice" / "lib",
            home / "Documents" / "LTspice" / "lib",
            Path("/Applications/LTspice.app/Contents/Resources/lib"),
            Path("/Applications/LTspice.app/Contents/lib"),
        ]
    elif "windows" in sysname:
        user = Path(os.environ.get("USERPROFILE", str(home)))
        pf   = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        pfx  = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        oned = Path(os.environ.get("OneDrive", "")) if os.environ.get("OneDrive") else None
        c += [
            user / "Documents" / "LTspice" / "lib",
            user / "Documents" / "LTspiceXVII" / "lib",
            pf / "LTspice" / "lib",
            pf / "Analog Devices" / "LTspice" / "lib",
            pfx / "LTC" / "LTspiceIV" / "lib",
        ]
        if oned:
            c += [oned / "Documents" / "LTspice" / "lib", oned / "Documents" / "LTspiceXVII" / "lib"]
    else:
        # Linux (Wine) and a common native spot
        wine = Path(os.environ.get("WINEPREFIX", str(home / ".wine")))
        user = os.environ.get("USERNAME", os.environ.get("USER", "Public"))
        c += [
            wine / "drive_c" / "users" / user / "Documents" / "LTspice" / "lib",
            wine / "drive_c" / "users" / "Public" / "Documents" / "LTspice" / "lib",
            home / "Documents" / "LTspice" / "lib",
        ]

    # de-dup, keep order
    seen, out = set(), []
    for p in c:
        p = p.resolve()
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def find_model(fullname):
    """Search for basename in known LTspice lib locations."""
    # Corpus netlists can retain Windows-style paths such as
    # `C:\\Users\\...\\standard.dio`. On macOS/Linux, os.path.basename()
    # does not treat backslashes as separators, so normalize first.
    fname = os.path.basename(fullname.replace("\\", "/"))
    for base in ltspice_lib_candidates():
        for subdir in ("sub", "", "cmp", "subckt"):
            cand = (base / subdir / fname)
            if cand.exists():
                return str(cand.resolve())
        # slow search fallback within this base
        try:
            return str(next(base.rglob(fname)).resolve())
        except StopIteration:
            pass
    return None

def replace_lib_paths(net_text):
    """
    Replace each .lib line's path with absolute path to the found file.
    Handles quoted and unquoted paths.
    """
    # Matches: .lib <path>   where <path> may be "quoted path" or bare
    pattern = re.compile(r'(^\s*\.lib\s+)(?P<path>"[^"]+"|\S+)', re.IGNORECASE | re.MULTILINE)

    def repl(m):
        prefix = m.group(1)
        raw = m.group('path')
        # strip quotes if present
        path_token = raw[1:-1] if raw.startswith('"') and raw.endswith('"') else raw
        # basename is what we search for
        target = find_model(path_token)
        if target:
            return f'{prefix}"{target}"'
        # if not found, keep original
        return m.group(0)

    return pattern.sub(repl, net_text)



def find_ltspice_executable() -> Optional[str]:
    """Find LTspice executable on the system."""
    logger.info("=== Finding LTspice executable ===")
    ENV = "LTSPICE_CMD"
    env_cmd = os.getenv(ENV)
    logger.debug(f"Environment variable {ENV} = {env_cmd}")
    
    def _is_exe(p: Path) -> bool:
        result = p.exists() and p.is_file() and os.access(str(p), os.X_OK)
        logger.debug(f"  Checking if {p} is executable: {result}")
        return result
    
    if env_cmd:
        logger.debug(f"Checking LTSPICE_CMD: {env_cmd}")
        p = Path(env_cmd.strip('"'))
        if _is_exe(p):
            logger.info(f"Found via LTSPICE_CMD: {p}")
            return str(p)
        w = shutil.which(env_cmd)
        if w:
            logger.info(f"Found via which(LTSPICE_CMD): {w}")
            return w
    
    logger.debug("Searching PATH for LTspice candidates...")
    for name in ["ltspice", "LTspice", "XVIIx64", "ltspice64"]:
        w = shutil.which(name)
        logger.debug(f"  which({name}) = {w}")
        if w:
            logger.info(f"Found in PATH: {w}")
            return w
    
    if sys.platform == "darwin":
        logger.debug("Checking macOS application bundle...")
        app_dir = Path("/Applications/LTspice.app")
        if app_dir.exists():
            logger.debug(f"Found LTspice.app at {app_dir}")
            internal = app_dir / "Contents" / "MacOS" / "LTspice"
            if _is_exe(internal):
                logger.info(f"Found macOS internal executable: {internal}")
                return str(internal)
            logger.info("Using 'open-app' method for macOS")
            return "open-app"
    
    if sys.platform.startswith("win"):
        logger.debug("Checking Windows default locations...")
        for p in [
            r"C:\Program Files\Analog Devices\LTspice\LTspice.exe",
            r"C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe",
        ]:
            if _is_exe(Path(p)):
                logger.info(f"Found Windows executable: {p}")
                return p
    
    logger.error("LTspice executable not found!")
    return None

def build_ltspice_command(exe: str, netlist_path: Path, batch: bool = True) -> List[str]:
    """Build LTspice command line."""
    logger.info(f"=== Building LTspice command ===")
    logger.debug(f"  exe: {exe}")
    logger.debug(f"  netlist_path: {netlist_path}")
    logger.debug(f"  batch mode: {batch}")
    
    if exe == "open-app":
        if batch:
            cmd = ["open", "-W", "-a", "LTspice", "--args", "-b", str(netlist_path)]
        else:
            cmd = ["open", "-a", "LTspice", str(netlist_path)]
    else:
        if batch:
            cmd = [exe, "-b", str(netlist_path)]
        else:
            cmd = [exe, str(netlist_path)]
    
    logger.info(f"Command: {' '.join(cmd)}")
    return cmd

class SimulationRunner:
    """Manages LTspice simulation execution."""
    
    def __init__(self):
        logger.info("Initializing SimulationRunner")
        self.exe = find_ltspice_executable()
        self.process = None
        self.result = None

        # --- USE YOUR NEW LOGIC ---
        logger.debug("Attempting to find LTspice 'lib' directory...")
        
        # Call your function to find candidate paths
        lib_paths = ltspice_lib_candidates() 
        
        # Find the first one that actually exists
        lib_path = next((p for p in lib_paths if p.exists()), None)
        
        if lib_path:
            logger.info(f"Found LTspice 'lib' dir: {lib_path}")
            # The env var needs the 'cmp' subdirectory path
            cmp_cand = lib_path / "cmp"
            if cmp_cand.exists():
                self.lib_cmp_dir = cmp_cand
                logger.debug(f"Set cmp dir for env: {self.lib_cmp_dir}")
            else:
                logger.warning(f"'cmp' subdir not found in {lib_path}.")
                self.lib_cmp_dir = None
        else:
            logger.error("LTspice 'lib' directory not found by guesser.")
            self.lib_cmp_dir = None
        # --- END OF NEW LOGIC ---

        self.proc_env = self._build_process_env()
    
    def _build_process_env(self) -> Dict[str, str]:
        """Return subprocess env with LTspice library paths fixed."""
        env = os.environ.copy()
        
        # Use the lib_cmp_dir found during initialization
        if self.lib_cmp_dir and self.lib_cmp_dir.exists():
            lib_root = self.lib_cmp_dir.parent
            sym_dir = lib_root / "sym"
            
            env["LTCOMPDIR"] = str(self.lib_cmp_dir)
            logger.debug(f"Set LTCOMPDIR={self.lib_cmp_dir}")
            
            if sym_dir.exists():
                env.setdefault("LTSYMDIR", str(sym_dir))
            env.setdefault("LTSPICEROOT", str(lib_root))
        else:
            logger.debug("lib_cmp_dir not found; using inherited environment only.")
        
        logger.debug(f"Process env overrides: {{k: env[k] for k in ['LTCOMPDIR','LTSYMDIR','LTSPICEROOT'] if k in env}}")
        return env
    
    def is_available(self) -> bool:
        """Check if LTspice is available."""
        available = self.exe is not None
        logger.debug(f"is_available() = {available}")
        return available
    
    # def _fix_lib_paths_if_needed(self, netlist_path: Path) -> None:
    #     """
    #     Rewrite .lib lines that use bare filenames (e.g., 'standard.dio') or
    #     Windows absolute paths so batch CLI can locate them on macOS.
    #     """
    #     if not self.lib_cmp_dir:
    #         logger.debug("No cmp dir known; skipping .lib fix.")
    #         return
    #     try:
    #         text = netlist_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    #     except Exception as e:
    #         logger.debug(f"Cannot read netlist for .lib fix: {e}")
    #         return
        
    #     changed = False
    #     new_lines = []
    #     for ln in text:
    #         low = ln.strip().lower()
    #         if low.startswith(".lib"):
    #             # tokens: .lib <path-or-file>
    #             parts = ln.split()
    #             if len(parts) >= 2:
    #                 target = parts[1].strip('"')
    #                 original_target = target
    #                 # If Windows path -> replace filename only using cmp dir
    #                 if "\\" in target and ":" in target:
    #                     fname = Path(target).name
    #                     candidate = self.lib_cmp_dir / fname
    #                     if candidate.exists():
    #                         target = str(candidate)
    #                         logger.debug(f"Rewriting Windows .lib path '{original_target}' -> '{target}'")
    #                         changed = True
    #                 # Bare filename (no slash) -> try cmp dir
    #                 elif ("/" not in target) and (not Path(target).is_absolute()):
    #                     candidate = self.lib_cmp_dir / target
    #                     if candidate.exists():
    #                         logger.debug(f"Expanding bare .lib '{target}' -> '{candidate}'")
    #                         target = str(candidate)
    #                         changed = True
    #                 # If we changed target rebuild line
    #                 if target != original_target:
    #                     # preserve original spacing (simple rebuild)
    #                     new_lines.append(f".lib {target}")
    #                     continue
    #         new_lines.append(ln)
        
    #     if changed:
    #         try:
    #             netlist_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    #             logger.info(f".lib directives rewritten for: {netlist_path.name}")
    #         except Exception as e:
    #             logger.error(f"Failed to write updated netlist (.lib fix): {e}")
    #     else:
    #         logger.debug("No .lib rewrite needed.")
    
    def run_batch(self, netlist_path: Path, timeout: int = 180) -> Dict[str, object]:
        """Run simulation in batch mode (blocking)."""
        logger.info(f"=== Starting batch simulation ===")
        logger.info(f"Netlist: {netlist_path}")
        logger.info(f"Timeout: {timeout}s")
        
        if not self.exe:
            logger.error("Cannot run: LTspice not found")
            return {"ok": False, "error": "LTspice not found"}
        
        netlist_path = netlist_path.resolve()
        logger.debug(f"Resolved netlist path: {netlist_path}")
        
        if not netlist_path.exists():
            logger.error(f"Netlist file does not exist: {netlist_path}")
            return {"ok": False, "error": f"Netlist file not found: {netlist_path}"}
        
        logger.debug(f"Netlist file size: {netlist_path.stat().st_size} bytes")
        logger.debug(f"Working directory: {netlist_path.parent}")
        
        # --- NEW: Use your function to rewrite .lib paths ---
        try:
            logger.debug(f"Reading netlist {netlist_path} for .lib rewrite...")
            original_text = netlist_path.read_text(encoding="utf-8", errors="ignore")
            
            # Call your new function
            fixed_text = replace_lib_paths(original_text) 
            
            if original_text != fixed_text:
                netlist_path.write_text(fixed_text, encoding="utf-8")
                logger.info(".lib paths rewritten with absolute paths.")
            else:
                logger.debug("No .lib paths needed rewriting.")
        except Exception as e:
            logger.error(f"Failed to read/rewrite netlist .lib paths: {e}", exc_info=True)
            # We can still try to run, in case it wasn't fatal
        # --- End new section ---
        
        cmd = build_ltspice_command(self.exe, netlist_path, batch=True)
        
        logger.info("Executing LTspice...")
        start = time.time()
        try:
            logger.debug("Running subprocess...")
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                cwd=netlist_path.parent,
                text=True,
                env=self.proc_env,
            )
            logger.info(f"Process completed with return code: {proc.returncode}")
            
        except subprocess.TimeoutExpired:
            logger.error(f"Process timed out after {timeout}s")
            return {"ok": False, "error": f"Timeout after {timeout}s"}
        except FileNotFoundError as e:
            logger.error(f"Executable not found: {e}")
            return {"ok": False, "error": f"Executable not found: {e}"}
        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            return {"ok": False, "error": f"Execution failed: {e}"}
        
        duration = time.time() - start
        logger.info(f"Execution completed in {duration:.3f}s")
        
        # (The rest of the function remains the same as the original)
        
        if proc.stdout:
            logger.debug(f"STDOUT:\n{proc.stdout}")
        if proc.stderr:
            logger.debug(f"STDERR:\n{proc.stderr}")
        
        raw_path = netlist_path.with_suffix(".raw")
        log_path = netlist_path.with_suffix(".log")
        
        logger.debug(f"Checking for .raw file: {raw_path}")
        raw_exists = raw_path.exists()
        logger.info(f"RAW file exists: {raw_exists}")
        if raw_exists:
            logger.debug(f"RAW file size: {raw_path.stat().st_size} bytes")
        
        logger.debug(f"Checking for .log file: {log_path}")
        log_exists = log_path.exists()
        logger.info(f"LOG file exists: {log_exists}")
        if log_exists:
            logger.debug(f"LOG file size: {log_path.stat().st_size} bytes")
            try:
                log_content = log_path.read_text(encoding="utf-8", errors="ignore")
                logger.debug(f"LOG content (first 500 chars):\n{log_content[:500]}")
                logger.debug(f"LOG content (last 500 chars):\n{log_content[-500:]}")
            except Exception as e:
                logger.error(f"Could not read log file: {e}")
        
        sim_ok = proc.returncode == 0 and raw_exists
        logger.info(f"Simulation success: {sim_ok} (returncode={proc.returncode}, raw_exists={raw_exists})")
        
        result = {
            "ok": sim_ok,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "netlist_path": str(netlist_path),
            "raw_path": str(raw_path) if raw_exists else None,
            "log_path": str(log_path) if log_exists else None,
            "elapsed_sec": round(duration, 3),
        }
        
        if not sim_ok:
            if not raw_exists:
                error_msg = "Simulation failed: No .raw file generated"
                logger.error(error_msg)
                result["error"] = error_msg
                
                if log_exists:
                    try:
                        log_content = log_path.read_text(encoding="utf-8", errors="ignore")
                        logger.debug("Searching for errors in log file...")
                        for line in log_content.splitlines():
                            if "error" in line.lower() or "fatal" in line.lower():
                                logger.warning(f"Found error in log: {line.strip()}")
                                result["error"] = line.strip()
                                break
                    except Exception as e:
                        logger.error(f"Could not parse log for errors: {e}")
            elif proc.returncode != 0:
                error_msg = f"LTspice exited with code {proc.returncode}"
                logger.error(error_msg)
                result["error"] = error_msg
        
        logger.info(f"=== Simulation result: {'SUCCESS' if sim_ok else 'FAILED'} ===")
        return result
    
    def run_async(self, netlist_path: Path, callback=None):
        """Run simulation in background thread."""
        logger.info(f"Starting async simulation for: {netlist_path}")
        
        def _run():
            logger.debug("Async thread started")
            self.result = self.run_batch(netlist_path)
            logger.debug("Async simulation completed, calling callback")
            if callback:
                callback(self.result)
            logger.debug("Callback completed")
        
        thread = Thread(target=_run, daemon=True)
        thread.start()
        logger.info(f"Async thread launched: {thread.name}")
        return thread