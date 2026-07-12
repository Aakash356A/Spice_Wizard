"""GUI application for SPICE netlist editing and simulation."""

import os

# PyLTSpice/matplotlib and torch can load separate OpenMP runtimes on macOS.
# This must be set before importing either stack, not only in ft_mac.py.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import Optional
import sys
import math
import logging
import threading
from datetime import datetime

from netlist_editor import NetlistModel, parse_netlist, set_element_value, set_param_value, set_analysis, save_netlist
from simulation_runner import SimulationRunner
from plot_manager import PlotManager

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# NEW import for Agent
try:
    from agent_core import UnifiedAgent
except ImportError:
    UnifiedAgent = None
    logger.warning("UnifiedAgent not available. Ensure agent_core.py is present.")

class SPICEEditorGUI:
    """Main GUI application for SPICE netlist editing."""
    
    def __init__(self, root):
        logger.info("Initializing SPICE Editor GUI")
        self.root = root
        self.root.title("SPICE Netlist Editor & Simulator")
        self.root.geometry("1200x800")
        
        self.netlist_path: Optional[Path] = None
        self.model: Optional[NetlistModel] = None
        self.output_path: Optional[Path] = None
        
        self.sim_runner = SimulationRunner()
        self.plot_manager = PlotManager()
        
        self.netlist_text_widget = None  # reference to new pane
        
        # The simulator and verifier should start instantly. Loading a local
        # foundation model is therefore an explicit opt-in rather than a GUI
        # startup side effect.
        self.chatbot = None
        self.agent_error = None
        local_agent_enabled = os.getenv("SPICE_WIZARD_ENABLE_LOCAL_AGENT", "").lower() in {
            "1", "true", "yes", "on"
        }
        api_configured = bool(os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY"))
        if UnifiedAgent is None:
            self.agent_error = "Agent dependencies are unavailable. Install requirements.txt."
        elif not (local_agent_enabled or api_configured):
            self.agent_error = (
                "No LLM backend configured. In .env set LLM_BASE_URL + LLM_API_KEY "
                "(AMD MI300X endpoint — see notebooks/amd_serve_and_tunnel.ipynb) or "
                "OPENROUTER_API_KEY. Optionally set SPICE_WIZARD_ENABLE_LOCAL_AGENT=1 "
                "for the local Gemma specialist after `git lfs pull`."
            )
        else:
            try:
                self.chatbot = UnifiedAgent()
            except Exception as e:
                self.agent_error = str(e)
                logger.warning("UnifiedAgent unavailable: %s", e)
        
        self.pending_llm_netlist: str | None = None
        # references for LLM widgets
        self.llm_conv_text = None
        self.llm_input_entry = None
        self.llm_send_btn = None
        self.llm_load_btn = None
        self.llm_status_label = None

        # Combine feature state
        self.combine_netlists: list[dict] = []  # [{'label':..., 'text':...}]
        self.combine_listbox = None
        self.combine_instr_text = None
        self.combine_status_label = None
        self.combine_merge_btn = None

        # Verify Spec feature state (Use Case 1: sim_harness.run_spice + spec_report.report)
        self.verify_metrics: list[dict] = []  # [{'name','target','tol','direction'}]

        # Candidate Arena state: multiple untrusted model outputs compete under
        # the same template invariants and LTspice measurement conditions.
        self.arena_candidates: list[dict[str, str]] = []
        self.arena_outcomes = []
        self.arena_outcome_items: dict[str, object] = {}
        self.arena_context: dict | None = None
        self.arena_candidate_listbox = None
        self.arena_results_tree = None
        self.arena_source_combo = None
        self.arena_status_label = None
        self.arena_run_btn = None
        self.arena_load_btn = None
        self.arena_export_btn = None

        self._create_widgets()
        self._check_dependencies()
        logger.info("GUI initialization complete")
    
    def _check_dependencies(self):
        """Check if required tools are available."""
        if not self.sim_runner.is_available():
            messagebox.showwarning(
                "LTspice Not Found",
                "LTspice executable not found. Simulation features will be disabled.\n\n"
                "Set LTSPICE_CMD environment variable or add ltspice to PATH."
            )
    
    def _create_widgets(self):
        """Create GUI widgets."""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Netlist...", command=self.open_netlist)
        file_menu.add_command(label="Save As...", command=self.save_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # File info
        info_frame = ttk.LabelFrame(main_frame, text="Netlist Info", padding="5")
        info_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.info_label = ttk.Label(info_frame, text="No netlist loaded")
        self.info_label.grid(row=0, column=0, sticky=tk.W)
        
        ttk.Button(info_frame, text="Open...", command=self.open_netlist).grid(row=0, column=1, padx=5)
        
        # Left panel: Elements editor
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Elements table
        elements_frame = ttk.LabelFrame(left_frame, text="Elements (R/C/L)", padding="5")
        elements_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.elements_tree = ttk.Treeview(
            elements_frame,
            columns=("Type", "Nodes", "Value"),
            show="tree headings",
            selectmode="browse"
        )
        self.elements_tree.heading("#0", text="Name")
        self.elements_tree.heading("Type", text="Type")
        self.elements_tree.heading("Nodes", text="Nodes")
        self.elements_tree.heading("Value", text="Value")
        
        self.elements_tree.column("#0", width=80)
        self.elements_tree.column("Type", width=50)
        self.elements_tree.column("Nodes", width=120)
        self.elements_tree.column("Value", width=100)
        
        vsb = ttk.Scrollbar(elements_frame, orient="vertical", command=self.elements_tree.yview)
        self.elements_tree.configure(yscrollcommand=vsb.set)
        
        self.elements_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.elements_tree.bind("<Double-1>", self.edit_element)
        
        # Params table
        params_frame = ttk.LabelFrame(left_frame, text="Parameters", padding="5")
        params_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.params_tree = ttk.Treeview(
            params_frame,
            columns=("Value",),
            show="tree headings",
            selectmode="browse"
        )
        self.params_tree.heading("#0", text="Name")
        self.params_tree.heading("Value", text="Value")
        
        self.params_tree.column("#0", width=150)
        self.params_tree.column("Value", width=150)
        
        vsb2 = ttk.Scrollbar(params_frame, orient="vertical", command=self.params_tree.yview)
        self.params_tree.configure(yscrollcommand=vsb2.set)
        
        self.params_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb2.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.params_tree.bind("<Double-1>", self.edit_param)
        
        # Right panel: Analysis & Simulation
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        main_frame.columnconfigure(1, weight=1)
        
        # Add notebook to hold Analysis / Netlist / Log / Plot panes
        nb = ttk.Notebook(right_frame)
        nb.pack(fill=tk.BOTH, expand=True)
        
        # --- Analysis tab (moved previous analysis + simulation controls here) ---
        analysis_tab = ttk.Frame(nb)
        nb.add(analysis_tab, text="Analysis & Run")
        
        analysis_frame = ttk.LabelFrame(analysis_tab, text="Analysis", padding="5")
        analysis_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(analysis_frame, text="Type:").grid(row=0, column=0, sticky=tk.W)
        self.analysis_type = ttk.Combobox(
            analysis_frame,
            values=[".ac", ".tran", ".dc"],
            state="readonly",
            width=10
        )
        self.analysis_type.grid(row=0, column=1, sticky=tk.W, padx=5)
        self.analysis_type.set(".ac")
        
        ttk.Label(analysis_frame, text="Arguments:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.analysis_args = ttk.Entry(analysis_frame, width=40)
        self.analysis_args.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5)
        self.analysis_args.insert(0, "dec 20 1 1e6")
        
        ttk.Button(analysis_frame, text="Update Analysis", command=self.update_analysis).grid(
            row=2, column=0, columnspan=3, pady=5
        )
        
        # Simulation controls
        sim_frame = ttk.LabelFrame(analysis_tab, text="Simulation", padding="5")
        sim_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            sim_frame,
            text="Save & Run Simulation",
            command=self.run_simulation
        ).pack(fill=tk.X, pady=2)
        
        self.status_label = ttk.Label(sim_frame, text="Ready", foreground="green")
        self.status_label.pack(fill=tk.X, pady=5)
        
        # --- Plot tab ---
        plot_tab = ttk.Frame(nb)
        nb.add(plot_tab, text="Plot Nodes")
        
        # --- Frame for plotting from list ---
        plot_frame = ttk.LabelFrame(plot_tab, text="Plot Nodes from List", padding="5")
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(plot_frame, text="Select voltage nodes to plot:").pack(anchor=tk.W)
        
        self.plot_listbox = tk.Listbox(plot_frame, selectmode=tk.MULTIPLE, height=8)
        self.plot_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # --- Frame for building expressions ---
        build_expr_frame = ttk.Frame(plot_frame)
        build_expr_frame.pack(fill=tk.X, pady=2)
        ttk.Button(build_expr_frame, text="Add Selected Node to Expression", command=self.add_node_to_expression).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(plot_frame, text="Generate Plot from Selection", command=self.generate_plot).pack(fill=tk.X)

        # --- Frame for plotting an expression ---
        expr_frame = ttk.LabelFrame(plot_tab, text="Plot Expression", padding="5")
        expr_frame.pack(fill=tk.X, pady=5)

        # Preset expressions
        presets_frame = ttk.Frame(expr_frame)
        presets_frame.pack(fill=tk.X, pady=(2,4))
        ttk.Label(presets_frame, text="Presets:").pack(side=tk.LEFT, padx=(0,5))
        self.preset_exprs = {
            "Gain (dB)": "(V(out)/V(in))",
            "Phase (deg)": "PHASE(V(out)/V(in))",
            "Group Delay": "GROUP_DELAY(V(out)/V(in))",
            "Output Impedance": "V(out)/I(Rload)",
            "Input Impedance": "V(in)/I(Vin)",
            "Load Power": "real(V(out)*I(Rload))",
            "Output Ripple": "max(V(out))-min(V(out))",
            "Efficiency": "(V(out)*I(Rload))/(V(in)*I(Vin))",
        }
        self.preset_combo = ttk.Combobox(presets_frame, values=list(self.preset_exprs.keys()), state="readonly", width=20)
        self.preset_combo.pack(side=tk.LEFT)
        ttk.Button(presets_frame, text="Insert Preset", command=self._insert_preset_expression).pack(side=tk.LEFT, padx=5)

        # Operator buttons
        op_frame = ttk.Frame(expr_frame)
        op_frame.pack(fill=tk.X, pady=(2, 5))
        ttk.Label(op_frame, text="Operators:").pack(side=tk.LEFT, padx=(0, 5))
        for op in ['+', '-', '*', '/', '(', ')']:
            ttk.Button(op_frame, text=op, width=3, command=lambda o=op: self.add_to_expression(f" {o} ")).pack(side=tk.LEFT)
        
        ttk.Button(op_frame, text="Clear", command=self.clear_expression, width=6).pack(side=tk.RIGHT)

        ttk.Label(expr_frame, text="Enter/Build expression (e.g., V(out)/V(n001)):").pack(anchor=tk.W)
        self.plot_expr_entry = ttk.Entry(expr_frame, width=50)
        self.plot_expr_entry.pack(fill=tk.X, pady=(2, 5))
        ttk.Button(expr_frame, text="Plot Expression", command=self.generate_expression_plot).pack(fill=tk.X)
        
        # --- Netlist tab (new) ---
        netlist_tab = ttk.Frame(nb)
        nb.add(netlist_tab, text="Netlist")
        nl_frame = ttk.LabelFrame(netlist_tab, text="Netlist Text (editable)", padding="5")
        nl_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.netlist_text_widget = scrolledtext.ScrolledText(nl_frame, wrap=tk.NONE, height=20)
        self.netlist_text_widget.pack(fill=tk.BOTH, expand=True)
        btn_bar = ttk.Frame(nl_frame)
        btn_bar.pack(fill=tk.X, pady=5)
        ttk.Button(btn_bar, text="Refresh from Model", command=self._refresh_netlist_text).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_bar, text="Apply Text Changes", command=self._apply_netlist_text_edits).pack(side=tk.LEFT, padx=4)
        
        # --- Log tab ---
        log_tab = ttk.Frame(nb)
        nb.add(log_tab, text="Log")
        log_frame = ttk.LabelFrame(log_tab, text="Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # ADD LLM TAB
        self._create_llm_tab(nb)
        self._create_combine_tab(nb)  # NEW
        self._create_verify_tab(nb)  # NEW: Use Case 1 spec verification
    
    def _create_llm_tab(self, nb):
        """Create ADI LLM chat tab."""
        llm_tab = ttk.Frame(nb)
        nb.add(llm_tab, text="ADI LLM")

        if not self.chatbot:
            message = "ADIChatbot unavailable."
            if self.agent_error:
                message += f"\n\nReason: {self.agent_error}"
            ttk.Label(llm_tab, text=message, justify=tk.CENTER, wraplength=500).pack(pady=20)
            return

        top_frame = ttk.LabelFrame(llm_tab, text="Conversation", padding=5)
        top_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.llm_conv_text = scrolledtext.ScrolledText(top_frame, wrap=tk.WORD, height=22)
        self.llm_conv_text.pack(fill=tk.BOTH, expand=True)
        self.llm_conv_text.insert(tk.END, "System ready. Ask for ADI ICs or modifications.\n")
        self.llm_conv_text.configure(state="disabled")

        input_frame = ttk.Frame(llm_tab)
        input_frame.pack(fill=tk.X, pady=5)

        self.llm_input_entry = ttk.Entry(input_frame)
        self.llm_input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.llm_input_entry.bind("<Return>", lambda e: self._llm_send())

        self.llm_send_btn = ttk.Button(input_frame, text="Send", command=self._llm_send)
        self.llm_send_btn.pack(side=tk.LEFT, padx=4)

        self.llm_load_btn = ttk.Button(input_frame, text="Load Netlist", command=self._llm_load_netlist, state="disabled")
        self.llm_load_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(input_frame, text="Print History", command=self._llm_print_history).pack(side=tk.LEFT, padx=4)

        ttk.Button(input_frame, text="Clear Chat", command=self._llm_clear_chat).pack(side=tk.LEFT, padx=4)

        self.llm_status_label = ttk.Label(llm_tab, text="Idle", foreground="green")
        self.llm_status_label.pack(fill=tk.X, pady=3)

    def _create_combine_tab(self, nb):
        """Create tab for multi-netlist combining via LLM."""
        combine_tab = ttk.Frame(nb)
        nb.add(combine_tab, text="Combine Circuits")
        
        top_frame = ttk.LabelFrame(combine_tab, text="Netlists to Merge", padding=5)
        top_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.combine_listbox = tk.Listbox(top_frame, height=10, selectmode=tk.SINGLE)
        self.combine_listbox.pack(fill=tk.BOTH, expand=True, pady=4)
        
        btn_row = ttk.Frame(top_frame)
        btn_row.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row, text="Add Current Editor", command=self._combine_add_editor).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="Add File...", command=self._combine_add_file).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="Add Last LLM Netlist", command=self._combine_add_last_llm).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="Remove Selected", command=self._combine_remove_selected).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="Clear All", command=self._combine_clear_all).pack(side=tk.LEFT, padx=3)
        
        instr_frame = ttk.LabelFrame(combine_tab, text="Connection / Mapping Instructions", padding=5)
        instr_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.combine_instr_text = scrolledtext.ScrolledText(instr_frame, height=8, wrap=tk.WORD)
        self.combine_instr_text.pack(fill=tk.BOTH, expand=True)
        self.combine_instr_text.insert(tk.END,"Merge Request-"
        )
        
        action_frame = ttk.Frame(combine_tab)
        action_frame.pack(fill=tk.X, pady=5)
        self.combine_merge_btn = ttk.Button(action_frame, text="Send Merge Request", command=self._combine_send_merge, state="disabled")
        self.combine_merge_btn.pack(side=tk.LEFT, padx=4)
        # store hidden load button
        self.combine_load_btn = ttk.Button(action_frame, text="Load Merged -> Editor", command=self._combine_load_merged, state="disabled")
        self.combine_load_btn.pack_forget()
        self.combine_status_label = ttk.Label(combine_tab, text="No netlists queued", foreground="gray")
        self.combine_status_label.pack(fill=tk.X, pady=3)

    def _create_verify_tab(self, nb):
        """Create the 'Verify Spec' tab -- Use Case 1: simulate the current
        netlist through LTspice and report measured metrics vs. target spec
        (sim_harness.run_spice + spec_report.report)."""
        verify_tab = ttk.Frame(nb)
        nb.add(verify_tab, text="Verify Spec")

        # --- Target metrics list ---
        metrics_frame = ttk.LabelFrame(verify_tab, text="Target Metrics", padding="5")
        metrics_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.verify_metrics_tree = ttk.Treeview(
            metrics_frame,
            columns=("Target", "Tol", "Direction"),
            show="tree headings",
            selectmode="browse",
            height=5,
        )
        self.verify_metrics_tree.heading("#0", text="Metric")
        self.verify_metrics_tree.heading("Target", text="Target")
        self.verify_metrics_tree.heading("Tol", text="Tol")
        self.verify_metrics_tree.heading("Direction", text="Direction")
        self.verify_metrics_tree.column("#0", width=110)
        self.verify_metrics_tree.column("Target", width=80)
        self.verify_metrics_tree.column("Tol", width=80)
        self.verify_metrics_tree.column("Direction", width=110)
        self.verify_metrics_tree.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        add_row = ttk.Frame(metrics_frame)
        add_row.pack(fill=tk.X, pady=2)
        ttk.Label(add_row, text="Name:").pack(side=tk.LEFT)
        self.verify_name_entry = ttk.Entry(add_row, width=12)
        self.verify_name_entry.pack(side=tk.LEFT, padx=3)
        self.verify_name_entry.insert(0, "gain_db")

        ttk.Label(add_row, text="Target:").pack(side=tk.LEFT)
        self.verify_target_entry = ttk.Entry(add_row, width=8)
        self.verify_target_entry.pack(side=tk.LEFT, padx=3)
        self.verify_target_entry.insert(0, "6.0")

        ttk.Label(add_row, text="Tol:").pack(side=tk.LEFT)
        self.verify_tol_entry = ttk.Entry(add_row, width=8)
        self.verify_tol_entry.pack(side=tk.LEFT, padx=3)
        self.verify_tol_entry.insert(0, "1.0")

        ttk.Label(add_row, text="Direction:").pack(side=tk.LEFT)
        self.verify_direction_combo = ttk.Combobox(
            add_row, values=["hit_target", "meet_at_least", "stay_below"],
            state="readonly", width=12
        )
        self.verify_direction_combo.pack(side=tk.LEFT, padx=3)
        self.verify_direction_combo.set("hit_target")

        ttk.Button(add_row, text="Add", command=self._verify_add_metric).pack(side=tk.LEFT, padx=3)
        ttk.Button(add_row, text="Remove Selected", command=self._verify_remove_metric).pack(side=tk.LEFT, padx=3)

        # --- Measurement node overrides (passed through to measure_raw.py) ---
        nodes_frame = ttk.LabelFrame(verify_tab, text="Measurement Nodes (for gain_db / bandwidth_hz)", padding="5")
        nodes_frame.pack(fill=tk.X, pady=5)
        ttk.Label(nodes_frame, text="In node:").grid(row=0, column=0, sticky=tk.W)
        self.verify_in_node_entry = ttk.Entry(nodes_frame, width=10)
        self.verify_in_node_entry.grid(row=0, column=1, padx=5)
        self.verify_in_node_entry.insert(0, "IN")

        ttk.Label(nodes_frame, text="Out node:").grid(row=0, column=2, sticky=tk.W)
        self.verify_out_node_entry = ttk.Entry(nodes_frame, width=10)
        self.verify_out_node_entry.grid(row=0, column=3, padx=5)
        self.verify_out_node_entry.insert(0, "OUT")

        ttk.Label(nodes_frame, text="Freq (Hz):").grid(row=0, column=4, sticky=tk.W)
        self.verify_freq_entry = ttk.Entry(nodes_frame, width=10)
        self.verify_freq_entry.grid(row=0, column=5, padx=5)
        self.verify_freq_entry.insert(0, "1000")

        # --- Run + results ---
        run_frame = ttk.Frame(verify_tab)
        run_frame.pack(fill=tk.X, pady=5)
        ttk.Button(run_frame, text="Run Verification", command=self._verify_run).pack(side=tk.LEFT)
        self.verify_status_label = ttk.Label(run_frame, text="Ready", foreground="green")
        self.verify_status_label.pack(side=tk.LEFT, padx=10)

        results_frame = ttk.LabelFrame(verify_tab, text="Results", padding="5")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.verify_results_tree = ttk.Treeview(
            results_frame,
            columns=("Target", "Measured", "Status", "Margin"),
            show="tree headings",
            selectmode="browse",
            height=6,
        )
        self.verify_results_tree.heading("#0", text="Metric")
        self.verify_results_tree.heading("Target", text="Target")
        self.verify_results_tree.heading("Measured", text="Measured")
        self.verify_results_tree.heading("Status", text="Status")
        self.verify_results_tree.heading("Margin", text="Margin %")
        self.verify_results_tree.column("#0", width=110)
        self.verify_results_tree.column("Target", width=80)
        self.verify_results_tree.column("Measured", width=80)
        self.verify_results_tree.column("Status", width=70)
        self.verify_results_tree.column("Margin", width=90)
        self.verify_results_tree.pack(fill=tk.BOTH, expand=True)

        # --- Best-of-N candidate arena ---
        arena_frame = ttk.LabelFrame(
            verify_tab,
            text="Candidate Arena — Best-of-N Verification",
            padding="5",
        )
        arena_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        ttk.Label(
            arena_frame,
            text=(
                "Import multiple AMD/local LLM responses. Each candidate must preserve the current "
                "editor template, then LTspice ranks it by measured specifications."
            ),
            wraplength=720,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 4))

        candidates_frame = ttk.Frame(arena_frame)
        candidates_frame.pack(fill=tk.X, pady=2)
        self.arena_candidate_listbox = tk.Listbox(candidates_frame, height=4, selectmode=tk.BROWSE)
        self.arena_candidate_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        candidate_buttons = ttk.Frame(candidates_frame)
        candidate_buttons.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(candidate_buttons, text="Add Candidate Files...", command=self._arena_add_candidates).pack(fill=tk.X)
        ttk.Button(candidate_buttons, text="Remove Selected", command=self._arena_remove_selected).pack(fill=tk.X, pady=2)
        ttk.Button(candidate_buttons, text="Clear", command=self._arena_clear_candidates).pack(fill=tk.X)

        arena_controls = ttk.Frame(arena_frame)
        arena_controls.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(arena_controls, text="Candidate source:").pack(side=tk.LEFT)
        self.arena_source_combo = ttk.Combobox(
            arena_controls,
            values=["amd_mi300x_manual", "local_fine_tune", "openai_compatible", "other"],
            state="readonly",
            width=22,
        )
        self.arena_source_combo.pack(side=tk.LEFT, padx=(4, 10))
        self.arena_source_combo.set("amd_mi300x_manual")
        self.arena_run_btn = ttk.Button(
            arena_controls,
            text="Run Candidate Arena",
            command=self._arena_run,
            state="disabled",
        )
        self.arena_run_btn.pack(side=tk.LEFT)
        self.arena_load_btn = ttk.Button(
            arena_controls,
            text="Load Selected Candidate",
            command=self._arena_load_selected,
            state="disabled",
        )
        self.arena_load_btn.pack(side=tk.LEFT, padx=4)
        self.arena_export_btn = ttk.Button(
            arena_controls,
            text="Export Evidence...",
            command=self._arena_export_evidence,
            state="disabled",
        )
        self.arena_export_btn.pack(side=tk.LEFT)
        self.arena_status_label = ttk.Label(arena_frame, text="Add candidates to begin", foreground="gray")
        self.arena_status_label.pack(fill=tk.X, pady=2)

        self.arena_results_tree = ttk.Treeview(
            arena_frame,
            columns=("Source", "Outcome", "Metrics", "Score", "Time"),
            show="tree headings",
            selectmode="browse",
            height=5,
        )
        self.arena_results_tree.heading("#0", text="Candidate")
        self.arena_results_tree.heading("Source", text="Source")
        self.arena_results_tree.heading("Outcome", text="Outcome")
        self.arena_results_tree.heading("Metrics", text="Metrics")
        self.arena_results_tree.heading("Score", text="Quality")
        self.arena_results_tree.heading("Time", text="Time")
        self.arena_results_tree.column("#0", width=180)
        self.arena_results_tree.column("Source", width=130)
        self.arena_results_tree.column("Outcome", width=110)
        self.arena_results_tree.column("Metrics", width=85)
        self.arena_results_tree.column("Score", width=80)
        self.arena_results_tree.column("Time", width=70)
        self.arena_results_tree.tag_configure("pass", foreground="green")
        self.arena_results_tree.tag_configure("fail", foreground="red")
        self.arena_results_tree.tag_configure("rejected", foreground="#a06000")
        self.arena_results_tree.pack(fill=tk.BOTH, expand=True)

    def _arena_refresh_candidates(self):
        """Refresh the candidate list and action availability."""
        if not self.arena_candidate_listbox:
            return
        self.arena_candidate_listbox.delete(0, tk.END)
        for candidate in self.arena_candidates:
            self.arena_candidate_listbox.insert(tk.END, candidate["label"])
        if self.arena_candidates:
            self.arena_status_label.config(
                text=f"{len(self.arena_candidates)} candidate(s) ready for LTspice verification",
                foreground="blue",
            )
            self.arena_run_btn.config(state="normal")
        else:
            self.arena_status_label.config(text="Add candidates to begin", foreground="gray")
            self.arena_run_btn.config(state="disabled")

    def _arena_add_candidates(self):
        """Import complete LLM responses or raw netlists into the arena."""
        paths = filedialog.askopenfilenames(
            title="Add Candidate Netlists or LLM Responses",
            filetypes=[
                ("Candidate files", "*.txt *.net *.cir *.sp *.spice"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return

        existing_labels = {candidate["label"] for candidate in self.arena_candidates}
        for raw_path in paths:
            path = Path(raw_path)
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as error:
                messagebox.showerror("Candidate Import Error", f"Could not read {path.name}:\n{error}")
                continue
            if not text.strip():
                messagebox.showwarning("Candidate Import", f"Skipped empty file: {path.name}")
                continue
            label = path.name
            suffix = 2
            while label in existing_labels:
                label = f"{path.stem} ({suffix}){path.suffix}"
                suffix += 1
            existing_labels.add(label)
            self.arena_candidates.append({"label": label, "text": text})
        self._arena_refresh_candidates()

    def _arena_remove_selected(self):
        if not self.arena_candidate_listbox:
            return
        selection = self.arena_candidate_listbox.curselection()
        if not selection:
            return
        self.arena_candidates.pop(selection[0])
        self._arena_refresh_candidates()

    def _arena_clear_candidates(self):
        self.arena_candidates.clear()
        self._arena_refresh_candidates()

    def _arena_run(self):
        """Run all imported candidates under one immutable template/spec context."""
        if not self.model:
            messagebox.showwarning("Candidate Arena", "Load the trusted template into the editor first.")
            return
        if not self.arena_candidates:
            messagebox.showwarning("Candidate Arena", "Add one or more candidate files first.")
            return
        if not self.verify_metrics:
            messagebox.showwarning("Candidate Arena", "Add at least one target metric first.")
            return
        if not self.sim_runner.is_available():
            messagebox.showerror("Candidate Arena", "LTspice not found.")
            return

        try:
            freq_hz = float(self.verify_freq_entry.get().strip() or "1000")
        except ValueError:
            messagebox.showerror("Candidate Arena", "Freq must be a number.")
            return
        if not math.isfinite(freq_hz) or freq_hz <= 0:
            messagebox.showerror("Candidate Arena", "Freq must be a finite number greater than zero.")
            return

        template_netlist = "\n".join(self.model.lines)
        candidate_snapshot = [dict(candidate) for candidate in self.arena_candidates]
        metrics_snapshot = [dict(metric) for metric in self.verify_metrics]
        in_node = self.verify_in_node_entry.get().strip() or "IN"
        out_node = self.verify_out_node_entry.get().strip() or "OUT"
        source = self.arena_source_combo.get() or "manual_candidate"
        template_label = self.netlist_path.name if self.netlist_path else "editor_template"
        template_id = self.netlist_path.stem if self.netlist_path else "GUI_TEMPLATE"
        spec_label = "GUI candidate arena: " + "; ".join(
            f"{metric['name']} {metric['direction']} {metric['target']} (tol {metric['tol']})"
            for metric in metrics_snapshot
        )

        self.arena_status_label.config(text="Verifying candidates in LTspice...", foreground="orange")
        self.arena_run_btn.config(state="disabled")
        self.log(f"Candidate Arena started with {len(candidate_snapshot)} candidate(s).")

        def worker():
            try:
                repo_root = str(Path(__file__).resolve().parent.parent)
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)
                from candidate_arena import CandidateInput, evaluate_candidate_batch, format_candidate_summary
                from generate_verify import log_verified_pair
                from sim_harness import Metric, Spec

                spec = Spec(
                    metrics=[
                        Metric(
                            name=metric["name"],
                            target=metric["target"],
                            tol=metric["tol"],
                            direction=metric["direction"],
                        )
                        for metric in metrics_snapshot
                    ],
                    testbench="",
                )
                outcomes = evaluate_candidate_batch(
                    template_netlist,
                    [
                        CandidateInput(label=candidate["label"], text=candidate["text"], source=source)
                        for candidate in candidate_snapshot
                    ],
                    spec,
                    in_node=in_node,
                    out_node=out_node,
                    freq_hz=freq_hz,
                )
                for outcome in outcomes:
                    if outcome.passed:
                        log_verified_pair(
                            template_id,
                            spec_label,
                            outcome.netlist,
                            outcome.reports,
                            attempt=outcome.rank or 1,
                            source=outcome.source,
                        )
                result = {
                    "outcomes": outcomes,
                    "summary": format_candidate_summary(outcomes),
                    "context": {
                        "template_label": template_label,
                        "template_netlist": template_netlist,
                        "spec_label": spec_label,
                        "metrics": metrics_snapshot,
                        "in_node": in_node,
                        "out_node": out_node,
                        "freq_hz": freq_hz,
                    },
                }
            except Exception as error:
                logger.error("Candidate Arena failed: %s", error, exc_info=True)
                result = {"error": str(error)}
            self.root.after(0, lambda: self._arena_handle_result(result))

        threading.Thread(target=worker, daemon=True).start()

    def _arena_handle_result(self, result: dict):
        """Render ranked Candidate Arena outcomes on the main Tk thread."""
        self.arena_run_btn.config(state="normal" if self.arena_candidates else "disabled")
        if result.get("error"):
            self.arena_status_label.config(text="Candidate Arena failed", foreground="red")
            self.log(f"Candidate Arena ERROR: {result['error']}")
            messagebox.showerror("Candidate Arena Failed", result["error"])
            return

        self.arena_outcomes = result["outcomes"]
        self.arena_context = result["context"]
        self.arena_outcome_items.clear()
        for item in self.arena_results_tree.get_children():
            self.arena_results_tree.delete(item)

        pass_count = 0
        for outcome in self.arena_outcomes:
            if outcome.passed:
                pass_count += 1
            tag = "pass" if outcome.passed else "rejected" if outcome.status == "REJECTED" else "fail"
            item = self.arena_results_tree.insert(
                "",
                tk.END,
                text=f"#{outcome.rank} {outcome.label}",
                values=(
                    outcome.source,
                    outcome.status,
                    f"{outcome.passed_metrics}/{outcome.total_metrics}",
                    f"{outcome.score:.3f}",
                    f"{outcome.elapsed_sec:.1f}s",
                ),
                tags=(tag,),
            )
            self.arena_outcome_items[item] = outcome

        self.arena_status_label.config(
            text=(
                f"{pass_count}/{len(self.arena_outcomes)} candidate(s) passed — "
                "highest ranked candidate is first"
            ),
            foreground="green" if pass_count else "red",
        )
        self.arena_load_btn.config(state="normal" if self.arena_outcomes else "disabled")
        self.arena_export_btn.config(state="normal" if self.arena_outcomes else "disabled")
        self.log(result["summary"])

    def _arena_load_selected(self):
        """Load the selected candidate into the editor for inspection or export."""
        selection = self.arena_results_tree.selection()
        if not selection:
            messagebox.showwarning("Candidate Arena", "Select a candidate result first.")
            return
        outcome = self.arena_outcome_items.get(selection[0])
        if outcome is None or not outcome.netlist.strip():
            messagebox.showwarning("Candidate Arena", "The selected result does not contain a usable netlist.")
            return
        try:
            model = parse_netlist(outcome.netlist)
            model.lines = outcome.netlist.splitlines()
            self.model = model
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.netlist_path = Path(f"arena_{outcome.label}_{stamp}.cir")
            self.info_label.config(text=f"Loaded: {self.netlist_path.name} ({outcome.status})")
            self._populate_ui()
            self.log(f"Loaded Candidate Arena result: {outcome.label} ({outcome.status}).")
        except Exception as error:
            messagebox.showerror("Candidate Arena", f"Could not load selected candidate:\n{error}")
            self.log(f"Candidate Arena load ERROR: {error}")

    def _arena_export_evidence(self):
        """Export a self-contained measurement/provenance record for a demo or review."""
        if not self.arena_outcomes or not self.arena_context:
            messagebox.showwarning("Candidate Arena", "Run the Candidate Arena before exporting evidence.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Candidate Arena Evidence",
            defaultextension=".json",
            filetypes=[("JSON evidence", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            repo_root = str(Path(__file__).resolve().parent.parent)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from candidate_arena import write_evidence_bundle
            from sim_harness import Metric, Spec

            context = self.arena_context
            spec = Spec(
                metrics=[
                    Metric(
                        name=metric["name"],
                        target=metric["target"],
                        tol=metric["tol"],
                        direction=metric["direction"],
                    )
                    for metric in context["metrics"]
                ],
                testbench="",
            )
            output_path = write_evidence_bundle(
                path,
                template_label=context["template_label"],
                template_netlist=context["template_netlist"],
                spec_label=context["spec_label"],
                spec=spec,
                in_node=context["in_node"],
                out_node=context["out_node"],
                freq_hz=context["freq_hz"],
                outcomes=self.arena_outcomes,
            )
            self.log(f"Candidate Arena evidence exported: {output_path}")
            messagebox.showinfo("Candidate Arena", f"Evidence exported to:\n{output_path}")
        except Exception as error:
            messagebox.showerror("Candidate Arena", f"Could not export evidence:\n{error}")
            self.log(f"Candidate Arena export ERROR: {error}")

    def _verify_add_metric(self):
        """Add one target metric row (used to build a sim_harness.Spec)."""
        name = self.verify_name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Metric name is required.")
            return
        if any(metric["name"].lower() == name.lower() for metric in self.verify_metrics):
            messagebox.showerror("Error", f"Metric '{name}' has already been added.")
            return
        try:
            target = float(self.verify_target_entry.get().strip())
            tol = float(self.verify_tol_entry.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Target and Tol must be numbers.")
            return
        if not math.isfinite(target) or not math.isfinite(tol) or tol < 0:
            messagebox.showerror("Error", "Target and Tol must be finite numbers; Tol must be zero or positive.")
            return
        direction = self.verify_direction_combo.get() or "hit_target"
        self.verify_metrics.append({"name": name, "target": target, "tol": tol, "direction": direction})
        self.verify_metrics_tree.insert("", tk.END, text=name, values=(target, tol, direction))

    def _verify_remove_metric(self):
        """Remove the selected metric row."""
        selection = self.verify_metrics_tree.selection()
        if not selection:
            return
        idx = self.verify_metrics_tree.index(selection[0])
        self.verify_metrics_tree.delete(selection[0])
        if 0 <= idx < len(self.verify_metrics):
            self.verify_metrics.pop(idx)

    def _verify_run(self):
        """Run Use Case 1 (sim_harness.run_spice + spec_report.report) on the
        current netlist, in a background thread (same pattern as run_simulation)."""
        if not self.model:
            messagebox.showwarning("Warning", "No netlist loaded.")
            return
        if not self.verify_metrics:
            messagebox.showwarning("Warning", "Add at least one target metric first.")
            return
        if not self.sim_runner.is_available():
            messagebox.showerror("Error", "LTspice not found.")
            return

        try:
            freq_hz = float(self.verify_freq_entry.get().strip() or "1000")
        except ValueError:
            messagebox.showerror("Error", "Freq must be a number.")
            return
        if not math.isfinite(freq_hz) or freq_hz <= 0:
            messagebox.showerror("Error", "Freq must be a finite number greater than zero.")
            return
        in_node = self.verify_in_node_entry.get().strip() or "IN"
        out_node = self.verify_out_node_entry.get().strip() or "OUT"

        netlist_text = "\n".join(self.model.lines)
        metrics_snapshot = list(self.verify_metrics)

        self.verify_status_label.config(text="Running verification...", foreground="orange")
        self.log("Starting Use Case 1 verification...")
        logger.info("Launching verification (run_spice) in background thread")

        def worker():
            try:
                # sim_harness.py / spec_report.py live at the repo root (one level
                # above app/); add it to sys.path so these bare imports resolve
                # regardless of how gui_main.py itself was launched.
                repo_root = str(Path(__file__).resolve().parent.parent)
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)
                from sim_harness import run_spice, Metric, Spec
                from spec_report import report as build_report

                metric_objs = [
                    Metric(name=m["name"], target=m["target"], tol=m["tol"], direction=m["direction"])
                    for m in metrics_snapshot
                ]
                sim = run_spice(
                    netlist_text,
                    testbench="",
                    in_node=in_node,
                    out_node=out_node,
                    freq_hz=freq_hz,
                    requested_measurements={m["name"] for m in metrics_snapshot},
                )
                if not sim.converged:
                    result = {"converged": False, "raw_log": sim.raw_log, "reports": [], "diagnostics": sim.diagnostics}
                else:
                    reports = build_report(sim, Spec(metrics=metric_objs, testbench=""))
                    result = {"converged": True, "raw_log": sim.raw_log, "reports": reports, "diagnostics": sim.diagnostics}
            except Exception as e:
                logger.error(f"Verification failed: {e}", exc_info=True)
                result = {"converged": False, "raw_log": f"Exception: {e}", "reports": [], "diagnostics": []}
            self.root.after(0, lambda: self._verify_handle_result(result))

        threading.Thread(target=worker, daemon=True).start()

    def _verify_handle_result(self, result: dict):
        """Handle verification result on the main thread."""
        for item in self.verify_results_tree.get_children():
            self.verify_results_tree.delete(item)

        if not result["converged"]:
            self.verify_status_label.config(text="Simulation failed", foreground="red")
            self.log(f"Verification ERROR: {result['raw_log']}")
            messagebox.showerror("Verification Failed", result["raw_log"] or "Simulation did not converge.")
            return

        all_passed = True
        for r in result["reports"]:
            status = "PASS" if r.passed else "FAIL"
            all_passed = all_passed and r.passed
            measured_str = "N/A" if math.isnan(r.measured) else f"{r.measured:.3f}"
            margin_str = "N/A" if math.isnan(r.margin_pct) else f"{r.margin_pct:+.1f}%"
            self.verify_results_tree.insert(
                "", tk.END, text=r.name,
                values=(r.target, measured_str, status, margin_str)
            )

        self.verify_status_label.config(
            text="All metrics PASS" if all_passed else "Some metrics FAILED",
            foreground="green" if all_passed else "red"
        )
        for diagnostic in result.get("diagnostics", []):
            self.log(f"Verification diagnostic: {diagnostic}")
        self.log(f"Verification complete: {'PASS' if all_passed else 'FAIL'}")

    def _llm_print_history(self):
        """Print LLM conversation history to the terminal."""
        if not self.chatbot:
            print("Chatbot not initialized.")
            return
        self.chatbot.show_history()
        
    def log(self, message: str):
        """Add message to log."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
    
    def open_netlist(self):
        """Open and parse a netlist file."""
        logger.info("Opening netlist file dialog")
        path = filedialog.askopenfilename(
            title="Open SPICE Netlist",
            filetypes=[
                ("SPICE Files", "*.cir *.sp *.spice *.net"),
                ("All Files", "*.*")
            ]
        )
        
        if not path:
            logger.debug("User cancelled file dialog")
            return
        
        self.netlist_path = Path(path)
        logger.info(f"Loading netlist: {self.netlist_path}")
        try:
            text = self.netlist_path.read_text(encoding="utf-8", errors="ignore")
            logger.debug(f"Netlist file size: {len(text)} characters")
            self.model = parse_netlist(text)
            self._populate_ui()
            self.info_label.config(text=f"Loaded: {self.netlist_path.name}")
            self.log(f"Loaded netlist: {self.netlist_path}")
            self.log(f"  Elements: {len(self.model.elements)}, Params: {len(self.model.params)}")
            logger.info(f"Successfully loaded netlist with {len(self.model.elements)} elements")
        except Exception as e:
            logger.error(f"Failed to load netlist: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load netlist:\n{e}")
            self.log(f"ERROR: {e}")
    
    def _refresh_netlist_text(self):
        """Write model.lines into the netlist text widget."""
        if not self.model or not self.netlist_text_widget:
            return
        self.netlist_text_widget.delete("1.0", tk.END)
        self.netlist_text_widget.insert(tk.END, "\n".join(self.model.lines) + "\n")
        self.log("Netlist pane refreshed from model.")
    
    def _apply_netlist_text_edits(self):
        """Parse edits from the netlist text widget and update the model."""
        if not self.netlist_text_widget:
            return
        raw = self.netlist_text_widget.get("1.0", tk.END)
        try:
            new_model = parse_netlist(raw)
        except Exception as e:
            messagebox.showerror("Parse Error", f"Failed to parse edited netlist:\n{e}")
            self.log(f"Parse ERROR: {e}")
            return
        # Keep lines exactly as edited
        new_model.lines = raw.rstrip("\n").splitlines()
        self.model = new_model
        self._populate_ui()  # refresh element / param trees
        self.log("Applied netlist text edits.")
    
    def _populate_ui(self):
        """Populate UI with netlist data."""
        if not self.model:
            return
        
        # Clear existing
        for item in self.elements_tree.get_children():
            self.elements_tree.delete(item)
        for item in self.params_tree.get_children():
            self.params_tree.delete(item)
        
        # Populate elements
        for name in sorted(self.model.elements.keys()):
            el = self.model.elements[name]
            self.elements_tree.insert(
                "",
                tk.END,
                text=el.name,
                values=(el.etype, ",".join(el.nodes), el.value_token or "")
            )
        
        # Populate params
        for name in sorted(self.model.params.keys()):
            p = self.model.params[name]
            self.params_tree.insert("", tk.END, text=p.key, values=(p.value_token,))
        
        # Update analysis
        if self.model.analyses:
            ac = self.model.analyses[-1]
            self.analysis_type.set(ac.kind)
            self.analysis_args.delete(0, tk.END)
            self.analysis_args.insert(0, ac.args)
        
        self._refresh_netlist_text()
    
    def edit_element(self, event):
        """Edit selected element value."""
        selection = self.elements_tree.selection()
        if not selection or not self.model:
            return
        
        item = selection[0]
        name = self.elements_tree.item(item, "text")
        current = self.elements_tree.item(item, "values")[2]
        
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit {name}")
        dialog.geometry("300x120")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text=f"Element: {name}").pack(pady=5)
        ttk.Label(dialog, text=f"Current value: {current}").pack()
        
        ttk.Label(dialog, text="New value:").pack(pady=5)
        entry = ttk.Entry(dialog, width=30)
        entry.pack(padx=10)
        entry.insert(0, current)
        entry.focus()
        
        def save():
            new_val = entry.get().strip()
            if new_val and set_element_value(self.model, name, new_val):
                self.elements_tree.item(item, values=(
                    self.elements_tree.item(item, "values")[0],
                    self.elements_tree.item(item, "values")[1],
                    new_val
                ))
                self.log(f"Updated {name} = {new_val}")
                dialog.destroy()
                self._refresh_netlist_text()
            else:
                messagebox.showerror("Error", "Failed to update element")
        
        ttk.Button(dialog, text="Save", command=save).pack(pady=10)
        entry.bind("<Return>", lambda e: save())
    
    def edit_param(self, event):
        """Edit selected parameter value."""
        selection = self.params_tree.selection()
        if not selection or not self.model:
            return
        
        item = selection[0]
        name = self.params_tree.item(item, "text")
        current = self.params_tree.item(item, "values")[0]
        
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit Parameter {name}")
        dialog.geometry("300x120")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text=f"Parameter: {name}").pack(pady=5)
        ttk.Label(dialog, text=f"Current value: {current}").pack()
        
        ttk.Label(dialog, text="New value:").pack(pady=5)
        entry = ttk.Entry(dialog, width=30)
        entry.pack(padx=10)
        entry.insert(0, current)
        entry.focus()
        
        def save():
            new_val = entry.get().strip()
            if new_val and set_param_value(self.model, name, new_val):
                self.params_tree.item(item, values=(new_val,))
                self.log(f"Updated param {name} = {new_val}")
                dialog.destroy()
                self._refresh_netlist_text()
            else:
                messagebox.showerror("Error", "Failed to update parameter")
        
        ttk.Button(dialog, text="Save", command=save).pack(pady=10)
        entry.bind("<Return>", lambda e: save())
    
    def update_analysis(self):
        """Update analysis card."""
        if not self.model:
            messagebox.showwarning("Warning", "No netlist loaded")
            return
        
        kind = self.analysis_type.get()
        args = self.analysis_args.get().strip()
        
        if not args:
            messagebox.showwarning("Warning", "Please enter analysis arguments")
            return
        
        try:
            set_analysis(self.model, kind, args)
            self.log(f"Updated analysis: {kind} {args}")
            messagebox.showinfo("Success", "Analysis updated")
            self._refresh_netlist_text()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update analysis:\n{e}")
            self.log(f"ERROR: {e}")
    
    def save_as(self):
        """Save netlist to a new file."""
        if not self.model:
            messagebox.showwarning("Warning", "No netlist loaded")
            return
        
        path = filedialog.asksaveasfilename(
            title="Save Netlist As",
            defaultextension=".cir",
            filetypes=[("SPICE Files", "*.cir *.sp *.spice"), ("All Files", "*.*")]
        )
        
        if not path:
            return
        
        try:
            save_netlist(self.model, Path(path))
            self.output_path = Path(path)
            self.log(f"Saved to: {path}")
            messagebox.showinfo("Success", f"Netlist saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}")
            self.log(f"ERROR: {e}")
    
    def run_simulation(self):
        """Save and run simulation."""
        logger.info("=== Starting simulation request ===")
        if not self.model or not self.netlist_path:
            logger.warning("No netlist loaded")
            messagebox.showwarning("Warning", "No netlist loaded")
            return
        
        if not self.sim_runner.is_available():
            logger.error("LTspice not available")
            messagebox.showerror("Error", "LTspice not found")
            return
        
        # Save netlist
        if not self.output_path:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = self.netlist_path.with_name(
                f"{self.netlist_path.stem}_edited_{ts}.cir"
            )
        
        logger.info(f"Saving netlist to: {self.output_path}")
        try:
            save_netlist(self.model, self.output_path)
            self.log(f"Saved to: {self.output_path}")
            logger.info("Netlist saved successfully")
        except Exception as e:
            logger.error(f"Failed to save netlist: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to save:\n{e}")
            return
        
        # Run simulation
        self.status_label.config(text="Running simulation...", foreground="orange")
        self.log("Starting LTspice simulation...")
        logger.info("Launching async simulation")
        
        def on_complete(result):
            logger.debug("Simulation callback triggered")
            # Schedule GUI updates on main thread
            self.root.after(0, lambda: self._handle_simulation_result(result))
        
        self.sim_runner.run_async(self.output_path, on_complete)
        logger.info("Async simulation launched")
    
    def _handle_simulation_result(self, result):
        """Handle simulation result on main thread."""
        logger.info("=== Processing simulation result ===")
        logger.debug(f"Result keys: {result.keys()}")
        logger.debug(f"Result ok: {result.get('ok')}")
        
        if result["ok"]:
            self.status_label.config(text="Simulation complete", foreground="green")
            self.log(f"Simulation completed in {result['elapsed_sec']}s")
            self.log(f"RAW file: {result.get('raw_path', 'N/A')}")
            logger.info(f"Simulation successful: {result.get('raw_path')}")
            
            # Load RAW and populate plot list
            if result.get("raw_path"):
                raw_path = Path(result["raw_path"])
                logger.info(f"Loading RAW file: {raw_path}")
                if self.plot_manager.load_raw(raw_path):
                    self._populate_plot_nodes()
                    self.log("RAW data loaded for plotting")
                    logger.info("RAW data loaded successfully")
                else:
                    self.log("Warning: Failed to parse RAW file")
                    logger.warning("Failed to parse RAW file")
        else:
            self.status_label.config(text="Simulation failed", foreground="red")
            error_msg = result.get("error", "Unknown error")
            self.log(f"ERROR: {error_msg}")
            logger.error(f"Simulation failed: {error_msg}")
            
            # Show detailed error if available
            if result.get("stderr"):
                error_msg += f"\n\nStderr:\n{result['stderr'][:500]}"
            
            if result.get("log_path"):
                try:
                    log_path = Path(result["log_path"])
                    if log_path.exists():
                        log_content = log_path.read_text(encoding="utf-8", errors="ignore")
                        logger.debug(f"Full log content:\n{log_content}")
                        error_msg += f"\n\nLog file (last 500 chars):\n{log_content[-500:]}"
                except Exception as e:
                    logger.error(f"Could not read log file: {e}")
            
            messagebox.showerror("Simulation Failed", error_msg)
    
    def _populate_plot_nodes(self):
        """Populate plot node listbox."""
        self.plot_listbox.delete(0, tk.END)
        for node in self.plot_manager.get_voltage_nodes():
            self.plot_listbox.insert(tk.END, node)
    
    def add_node_to_expression(self):
        """Appends the selected node from the listbox to the expression entry."""
        selection = self.plot_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a node from the list first.")
            return
        
        # We'll just add the first selected node if multiple are selected
        node_name = self.plot_listbox.get(selection[0])
        self.add_to_expression(node_name)

    def add_to_expression(self, text: str):
        """Inserts text at the cursor position in the expression entry."""
        self.plot_expr_entry.insert(tk.INSERT, text)
        self.plot_expr_entry.focus()

    def clear_expression(self):
        """Clears the expression entry field."""
        self.plot_expr_entry.delete(0, tk.END)

    def generate_plot(self):
        """Generate plot for selected nodes."""
        if not self.plot_manager.raw_data:
            messagebox.showwarning("Warning", "No simulation data available")
            return
        
        selection = self.plot_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select at least one node")
            return
        
        selected_nodes = [self.plot_listbox.get(i) for i in selection]
        self.log(f"Plotting: {', '.join(selected_nodes)}")
        
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Simulation Results")
        plot_window.geometry("800x600")
        
        try:
            fig = self.plot_manager.plot_nodes(selected_nodes)
            if fig and MATPLOTLIB_AVAILABLE:
                canvas = FigureCanvasTkAgg(fig, master=plot_window)
                canvas.draw()
                # Add toolbar for pan/zoom
                toolbar = NavigationToolbar2Tk(canvas, plot_window)
                toolbar.update()
                toolbar.pack(side=tk.TOP, fill=tk.X)
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                ttk.Label(plot_window, text="Matplotlib not available or plot failed.").pack()
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to generate plot:\n{e}")
            self.log(f"Plot ERROR: {e}")
            plot_window.destroy()

    def generate_expression_plot(self):
        """Generate plot for a user-defined expression."""
        if not self.plot_manager.raw_data:
            messagebox.showwarning("Warning", "No simulation data available")
            return

        expression = self.plot_expr_entry.get().strip()
        if not expression:
            messagebox.showwarning("Warning", "Please enter an expression to plot")
            return

        self.log(f"Plotting expression: {expression}")

        plot_window = tk.Toplevel(self.root)
        plot_window.title(f"Plot: {expression}")
        plot_window.geometry("800x600")

        try:
            fig = self.plot_manager.plot_expression(expression)
            if fig and MATPLOTLIB_AVAILABLE:
                canvas = FigureCanvasTkAgg(fig, master=plot_window)
                canvas.draw()
                toolbar = NavigationToolbar2Tk(canvas, plot_window)
                toolbar.update()
                toolbar.pack(side=tk.TOP, fill=tk.X)
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                messagebox.showerror("Plot Error", "Failed to generate plot for expression.")
                plot_window.destroy()
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot expression:\n{e}")
            self.log(f"Plot Expression ERROR: {e}")
            plot_window.destroy()

    def _llm_append(self, role: str, text: str):
        """Append text to conversation box."""
        if not self.llm_conv_text:
            return
        self.llm_conv_text.configure(state="normal")
        self.llm_conv_text.insert(tk.END, f"{role}: {text}\n")
        self.llm_conv_text.see(tk.END)
        self.llm_conv_text.configure(state="disabled")

    def _llm_clear_chat(self):
        if not self.chatbot:
            return
        self.chatbot.reset_conversation(keep_system=True)
        self.pending_llm_netlist = None
        self.llm_load_btn.configure(state="disabled")
        self.llm_conv_text.configure(state="normal")
        self.llm_conv_text.delete("1.0", tk.END)
        self.llm_conv_text.insert(tk.END, "Chat cleared. System ready.\n")
        self.llm_conv_text.configure(state="disabled")
        self.llm_status_label.config(text="Idle", foreground="green")

    def _llm_send(self):
        """Send user query to LLM asynchronously."""
        if not self.chatbot:
            return
        msg = self.llm_input_entry.get().strip()
        if not msg:
            return
        self.llm_input_entry.delete(0, tk.END)
        self._llm_append("You", msg)
        self.llm_status_label.config(text="Processing Request...", foreground="orange") # Changed text
        self.llm_send_btn.configure(state="disabled")

        def worker():
            try:
                # CHANGED: Use process_request instead of send_message
                result = self.chatbot.process_request(msg)
            except Exception as e:
                result = {"error": f"Exception: {e}"}
            self.root.after(0, lambda: self._llm_handle_result(result))

        threading.Thread(target=worker, daemon=True).start()

    def _combine_refresh_listbox(self):
        if not self.combine_listbox:
            return
        self.combine_listbox.delete(0, tk.END)
        for idx, item in enumerate(self.combine_netlists, start=1):
            lines = item['text'].count('\n') + 1
            self.combine_listbox.insert(tk.END, f"{idx}. {item['label']}  ({lines} lines)")
        if self.combine_netlists:
            self.combine_status_label.config(text=f"{len(self.combine_netlists)} netlists ready", foreground="blue")
            self.combine_merge_btn.configure(state="normal")
        else:
            self.combine_status_label.config(text="No netlists queued", foreground="gray")
            self.combine_merge_btn.configure(state="disabled")

    def _combine_add_editor(self):
        if not self.model:
            messagebox.showwarning("Warning", "No editor netlist loaded.")
            return
        text = "\n".join(self.model.lines)
        self.combine_netlists.append({"label": f"Editor:{self.netlist_path.name if self.netlist_path else 'Unsaved'}", "text": text})
        self._combine_refresh_listbox()

    def _combine_add_file(self):
        path = filedialog.askopenfilename(
            title="Add Netlist File",
            filetypes=[("SPICE Files", "*.cir *.sp *.spice *.net"), ("All Files", "*.*")]
        )
        if not path:
            return
        p = Path(path)
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file:\n{e}")
            return
        self.combine_netlists.append({"label": f"File:{p.name}", "text": txt})
        self._combine_refresh_listbox()

    def _combine_add_last_llm(self):
        if not self.chatbot or not self.chatbot.last_netlist:
            messagebox.showwarning("Warning", "No LLM netlist available.")
            return
        self.combine_netlists.append({"label": f"LLM:{self.chatbot.last_ic or 'NET'}", "text": self.chatbot.last_netlist})
        self._combine_refresh_listbox()

    def _combine_remove_selected(self):
        sel = self.combine_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.combine_netlists):
            self.combine_netlists.pop(idx)
        self._combine_refresh_listbox()

    def _combine_clear_all(self):
        self.combine_netlists.clear()
        self._combine_refresh_listbox()

    def _combine_send_merge(self):
        if not self.chatbot:
            messagebox.showerror("Error", "Chatbot unavailable.")
            return
        if len(self.combine_netlists) < 2:
            messagebox.showwarning("Warning", "Need at least two netlists to merge.")
            return
        instructions = self.combine_instr_text.get("1.0", tk.END).strip()
        if not instructions:
            messagebox.showwarning("Warning", "Provide mapping / connection instructions.")
            return
        self.combine_status_label.config(text="Merging via LLM...", foreground="orange")
        self.combine_merge_btn.configure(state="disabled")
        
        def worker():
            result = self.chatbot.merge_netlists(self.combine_netlists, instructions)
            self.root.after(0, lambda: self._combine_handle_merge_result(result))
        threading.Thread(target=worker, daemon=True).start()

    def _combine_handle_merge_result(self, result: dict):
        if "error" in result:
            self.combine_status_label.config(text=f"Merge error: {result['error']}", foreground="red")
            self.combine_merge_btn.configure(state="normal")
            return
        merged_netlist = result.get("netlist", "")
        if not merged_netlist:
            self.combine_status_label.config(text="LLM returned no merged netlist.", foreground="red")
            self.combine_merge_btn.configure(state="normal")
            return
        self.pending_llm_netlist = merged_netlist
        self.llm_load_btn.configure(state="normal")  # existing LLM load
        # show dedicated combine load button
        if hasattr(self, "combine_load_btn") and self.combine_load_btn:
            self.combine_load_btn.pack(side=tk.LEFT, padx=4)
            self.combine_load_btn.configure(state="normal")
        self.combine_status_label.config(text="Merged netlist ready (use Load buttons)", foreground="green")
        self.combine_merge_btn.configure(state="normal")
        if self.llm_conv_text:
            self._llm_append("LLM", f"[Merge Complete]\nSummary: {result.get('Summary','(no summary)')}")

    def _llm_handle_result(self, result: dict):
        """Handle LLM response on main thread."""
        self.llm_send_btn.configure(state="normal")
        
        # FIX: Check if error is truthy (not None), as the key often exists by default
        if result.get("error"):
            self._llm_append("LLM", f"Error: {result['error']}")
            self.llm_status_label.config(text="Error", foreground="red")
            return

        ic = result.get("IC", "N/A")
        mode = result.get("mode", "N/A")
        summary = result.get("Summary", "")
        netlist = result.get("netlist", "")

        self._llm_append("LLM", f"IC={ic} | mode={mode}\nSummary: {summary}")

        if mode == "merge" and netlist:
            # Treat merged netlist same as pending
            self.pending_llm_netlist = netlist
            self.llm_load_btn.configure(state="normal")
            self._llm_append("LLM", "[Merged netlist ready. Click 'Load Netlist'.]")
            self.llm_status_label.config(text="Merged ready", foreground="blue")
            return

        if netlist:
            self.pending_llm_netlist = netlist
            self.llm_load_btn.configure(state="normal")
            self._llm_append("LLM", "[Netlist ready. Click 'Load Netlist' to populate editor.]")
            self.llm_status_label.config(text="Netlist ready", foreground="blue")
        else:
            self.llm_status_label.config(text="Response received", foreground="green")

    def _llm_load_netlist(self):
        """Load the pending LLM (generated or merged) netlist into the editor."""
        if not self.pending_llm_netlist:
            messagebox.showwarning("Warning", "No pending LLM netlist to load.")
            return
        text = self.pending_llm_netlist.strip()
        if not text:
            messagebox.showwarning("Warning", "Pending LLM netlist is empty.")
            return
        try:
            model = parse_netlist(text)
            # Preserve exact text lines (including elements LLM may output outside parser scope)
            model.lines = text.splitlines()
            self.model = model
            # Virtual path (not saved yet)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.netlist_path = Path(f"llm_netlist_{ts}.cir")
            self.info_label.config(text=f"Loaded: {self.netlist_path.name} (LLM)")
            self._populate_ui()
            self.log("Loaded LLM netlist into editor.")
            self.llm_status_label.config(text="LLM netlist loaded", foreground="green")
            # After load, keep it available for further merge steps
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse LLM netlist:\n{e}")
            self.log(f"LLM load ERROR: {e}")

    def _combine_load_merged(self):
        """Load merged netlist (if pending) into editor (same logic as _llm_load_netlist)."""
        if not self.pending_llm_netlist:
            messagebox.showwarning("Warning", "No merged netlist to load.")
            return
        text = self.pending_llm_netlist.strip()
        if not text:
            messagebox.showwarning("Warning", "Merged netlist is empty.")
            return
        try:
            model = parse_netlist(text)
            model.lines = text.splitlines()
            self.model = model
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.netlist_path = Path(f"merged_netlist_{ts}.cir")
            self.info_label.config(text=f"Loaded: {self.netlist_path.name} (Merged)")
            self._populate_ui()
            self.log("Loaded merged netlist into editor.")
            self.combine_status_label.config(text="Merged netlist loaded", foreground="blue")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse merged netlist:\n{e}")
            self.log(f"Merged load ERROR: {e}")

    def _insert_preset_expression(self):
        """Insert selected preset expression."""
        if not hasattr(self, "preset_combo"):
            return
        key = self.preset_combo.get()
        if not key:
            return
        expr = self.preset_exprs.get(key, "")
        if expr:
            self.plot_expr_entry.delete(0, tk.END)
            self.plot_expr_entry.insert(0, expr)

def main():
    root = tk.Tk()
    app = SPICEEditorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()