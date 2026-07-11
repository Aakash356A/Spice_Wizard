"""
Plot manager for simulation results using PyLTSpice.
"""

from pathlib import Path
from typing import Optional, List, Dict
import logging
import numpy as np
import re
from matplotlib.ticker import EngFormatter

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available")

try:
    from PyLTSpice import RawRead
    PYLTSPICE_AVAILABLE = True
except ImportError:
    PYLTSPICE_AVAILABLE = False
    logging.warning("PyLTSpice not available. Install with: pip install PyLTSpice")

class PlotManager:
    """Manages plot generation from simulation results."""
    
    def __init__(self):
        self.available = MATPLOTLIB_AVAILABLE and PYLTSPICE_AVAILABLE
        self.raw_reader = None
        self.raw_path = None
        self.step_idx = 0  # For .STEP simulations
    
    @property
    def raw_data(self):
        """Compatibility property for GUI - returns True if raw_reader is loaded."""
        return self.raw_reader is not None
        
    def load_raw(self, raw_path: Path) -> bool:
        """Load .raw file data using PyLTSpice."""
        if not PYLTSPICE_AVAILABLE:
            logging.error("PyLTSpice not installed. Cannot load RAW file.")
            return False
            
        if not raw_path.exists():
            logging.error(f"RAW file does not exist: {raw_path}")
            return False
        
        try:
            self.raw_reader = RawRead(str(raw_path))
            self.raw_path = raw_path
            
            # Log available traces for debugging
            traces = self.raw_reader.get_trace_names()
            logging.info(f"Successfully loaded RAW file with {len(traces)} traces")
            logging.debug(f"Available traces: {traces}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to load RAW file {raw_path}: {e}", exc_info=True)
            self.raw_reader = None
            return False
    
    def get_voltage_nodes(self) -> List[str]:
        """Get list of voltage node names."""
        if not self.raw_reader:
            return []
        
        try:
            traces = self.raw_reader.get_trace_names()
            # Filter for voltage nodes (start with V()
            voltages = [n for n in traces if n.upper().startswith("V(")]
            return sorted(voltages)
        except Exception as e:
            logging.error(f"Failed to get voltage nodes: {e}")
            return []
    
    def get_all_traces(self) -> List[str]:
        """Get list of all trace names."""
        if not self.raw_reader:
            return []
        try:
            return sorted(self.raw_reader.get_trace_names())
        except Exception as e:
            logging.error(f"Failed to get traces: {e}")
            return []
    
    def _detect_analysis_type(self) -> tuple[str, np.ndarray]:
        """Detect if AC or TRAN analysis and return (axis_name, axis_data)."""
        traces = set(self.raw_reader.get_trace_names())
        
        if "frequency" in traces:
            axis_name = "frequency"
        elif "time" in traces:
            axis_name = "time"
        else:
            raise ValueError("No 'frequency' or 'time' axis found in RAW file")
        
        # Get axis data for the current step
        axis_trace = self.raw_reader.get_trace(axis_name)
        axis_data = axis_trace.get_wave(self.step_idx)
        
        return axis_name, axis_data
    
    def _apply_eng_format(self, ax, axis_name: str, is_ac: bool, y_kind: str):
        """
        Apply engineering SI prefixes to axes.
        axis_name: 'time' or 'frequency'
        is_ac: True if AC analysis
        y_kind: 'V','A','dB','value'
        """
        if axis_name == "time":
            ax.xaxis.set_major_formatter(EngFormatter(unit='s'))
        elif axis_name == "frequency":
            ax.xaxis.set_major_formatter(EngFormatter(unit='Hz'))

        if y_kind == "V":
            ax.yaxis.set_major_formatter(EngFormatter(unit='V'))
        elif y_kind == "A":
            ax.yaxis.set_major_formatter(EngFormatter(unit='A'))
        # For dB keep as-is; for generic value skip

    def plot_expression(self, expression: str, step_idx: int = 0) -> Optional[Figure]:
        """
        Parses and plots a mathematical expression of traces.
        Supports PHASE(<expr>) for phase (deg) and GROUP_DELAY(<expr>) for group delay.
        """
        if not self.available:
            logging.error("Plot manager not available.")
            return None
        if not self.raw_reader:
            raise ValueError("No RAW file loaded")

        self.step_idx = step_idx
        original_expr = expression.strip()

        is_phase = False
        is_gd = False
        # Detect wrappers
        phase_match = re.match(r"^PHASE\((.+)\)$", original_expr, re.IGNORECASE)
        gd_match = re.match(r"^GROUP_DELAY\((.+)\)$", original_expr, re.IGNORECASE)
        if phase_match:
            expression = phase_match.group(1)
            is_phase = True
        elif gd_match:
            expression = gd_match.group(1)
            is_gd = True
        
        trace_names = re.findall(r'[VI]\([^)]+\)', expression)
        if not trace_names:
            raise ValueError("Expression contains no valid traces to plot.")

        eval_context = {}
        safe_np = {
            'abs': np.abs, 'log10': np.log10, 'log': np.log, 'exp': np.exp,
            'sqrt': np.sqrt, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'real': np.real, 'imag': np.imag, 'conj': np.conj, 'pi': np.pi,
            'max': np.max, 'min': np.min, 'angle': np.angle
        }
        eval_context.update(safe_np)

        safe_expr = expression
        for i, name in enumerate(set(trace_names)):
            try:
                trace_data = self.raw_reader.get_trace(name).get_wave(self.step_idx)
                safe_name = f"trace{i}"
                eval_context[safe_name] = trace_data
                safe_expr = safe_expr.replace(name, safe_name)
            except Exception:
                raise ValueError(f"Trace '{name}' not found in simulation data.")

        try:
            result_data = eval(safe_expr, {"__builtins__": {}}, eval_context)
        except Exception as e:
            raise RuntimeError(f"Could not evaluate expression: {e}")

        axis_name, axis_data = self._detect_analysis_type()
        fig, ax = plt.subplots(figsize=(10, 6))

        if axis_name == "frequency":
            if is_phase:
                phase_deg = np.unwrap(np.angle(result_data)) * 180 / np.pi
                ax.semilogx(axis_data, phase_deg, label=original_expr)
                ax.set_ylabel("Phase (deg)")
                ax.set_title("AC Phase")
                self._apply_eng_format(ax, axis_name, True, "value")
            elif is_gd:
                # Group delay: -d(phi)/d(omega)
                omega = 2 * np.pi * axis_data
                phase_unwrapped = np.unwrap(np.angle(result_data))
                dphi = np.gradient(phase_unwrapped, omega)
                gd = -dphi
                ax.semilogx(axis_data, gd, label=original_expr)
                ax.set_ylabel("Group Delay (s)")
                ax.set_title("AC Group Delay")
                self._apply_eng_format(ax, axis_name, True, "value")
            else:
                mag_db = 20 * np.log10(np.abs(result_data))
                ax.semilogx(axis_data, mag_db, label=original_expr)
                ax.set_ylabel("Magnitude (dB)")
                ax.set_title("AC Analysis")
                self._apply_eng_format(ax, axis_name, True, "dB")
            ax.set_xlabel("Frequency (Hz)")
        else:
            # Transient
            ax.plot(axis_data, np.real(result_data), label=original_expr)
            ax.set_xlabel("Time (s)")
            if trace_names and all(t.upper().startswith("V(") for t in trace_names):
                y_lab = "Voltage (V)"; y_kind = "V"
            elif trace_names and all(t.upper().startswith("I(") for t in trace_names):
                y_lab = "Current (A)"; y_kind = "A"
            else:
                y_lab = "Value"; y_kind = "value"
            ax.set_ylabel(y_lab)
            ax.set_title("Transient Analysis")
            self._apply_eng_format(ax, axis_name, False, y_kind)

        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        logging.info(f"Successfully plotted expression: {original_expr}")
        return fig

    def plot_nodes(self, node_names: List[str], output_path: Optional[Path] = None,
                   step_idx: int = 0) -> Optional[Figure]:
        """Plot selected voltage nodes."""
        if not self.available:
            logging.error("Plot manager not available (matplotlib or PyLTSpice missing)")
            return None
            
        if not self.raw_reader:
            logging.error("No RAW file loaded")
            return None
        
        if not node_names:
            logging.warning("No node names provided for plotting")
            return None
        
        # Update step index
        self.step_idx = step_idx
        
        try:
            # Detect analysis type (AC or Transient)
            axis_name, axis_data = self._detect_analysis_type()
            logging.info(f"Detected analysis type: {axis_name}")
        except Exception as e:
            logging.error(f"Failed to detect analysis type: {e}", exc_info=True)
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted_count = 0
        
        if axis_name == "frequency":
            # AC Analysis
            for node in node_names:
                if node in self.raw_reader.get_trace_names():
                    try:
                        data = self.raw_reader.get_trace(node).get_wave(self.step_idx)
                        # Convert complex data to magnitude in dB
                        mag_db = 20 * np.log10(np.abs(data))
                        ax.semilogx(axis_data, mag_db, label=node)
                        plotted_count += 1
                    except Exception as e:
                        logging.error(f"Failed to plot node {node}: {e}")
                else:
                    logging.warning(f"Node {node} not found in traces")
                    
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.set_title("AC Analysis")
            self._apply_eng_format(ax, axis_name, True, "dB")
        else:
            # Transient Analysis
            for node in node_names:
                if node in self.raw_reader.get_trace_names():
                    try:
                        data = self.raw_reader.get_trace(node).get_wave(self.step_idx)
                        ax.plot(axis_data, np.real(data), label=node)
                        plotted_count += 1
                    except Exception as e:
                        logging.error(f"Failed to plot node {node}: {e}")
                else:
                    logging.warning(f"Node {node} not found in traces")
                    
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Voltage (V)")
            ax.set_title("Transient Analysis")
            self._apply_eng_format(ax, axis_name, False, "V")
        
        if plotted_count == 0:
            logging.error("No traces were successfully plotted")
            plt.close(fig)
            return None
        
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            try:
                fig.savefig(output_path, dpi=150, bbox_inches="tight")
                logging.info(f"Plot saved to: {output_path}")
            except Exception as e:
                logging.error(f"Failed to save plot: {e}")
        
        logging.info(f"Successfully plotted {plotted_count} traces")
        return fig