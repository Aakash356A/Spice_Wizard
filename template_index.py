"""
Scans the .net corpus once and builds {IC_NAME: [file_paths]} by extracting
the subcircuit name from each netlist's X... call lines.

Use Case 2, build step 1 (see USE_CASES_IMPLEMENTATION.md): given a part
name, this is how the generator retrieves a known-working template netlist
to adapt, instead of writing one from scratch.
"""

import glob
import os
import re

DEFAULT_NET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data", "templates")

# Tokens on an X line that are not the subcircuit name.
_PARAM_TOKEN = re.compile(r"^[\w.]+=\S+$")


def _subckt_name_from_x_line(line: str) -> str | None:
    """Return the subcircuit name from an LTspice `X...` element line.

    LTspice format: X<name> <node1> ... <nodeN> <SUBCKT_NAME> [param=value ...]
    The subcircuit name is the last token that isn't a param=value pair.
    Handles LTspice's unicode element prefix (e.g. `X§U1`).
    """
    tokens = line.split()
    if len(tokens) < 3:
        return None
    for tok in reversed(tokens[1:]):
        if not _PARAM_TOKEN.match(tok):
            return tok
    return None


def build_template_index(net_dir: str = DEFAULT_NET_DIR) -> dict[str, list[str]]:
    """{IC_NAME (upper): [netlist paths that instantiate it]}.

    The filename-matching heuristic is intentionally NOT used alone: a file
    named AD8092.net that never instantiates AD8092 would be a useless
    template. But when the file *is* named after the part it instantiates
    (the common case in this corpus), that path is listed first so callers
    can just take index[ic][0].
    """
    index: dict[str, list[str]] = {}
    for path in sorted(glob.glob(os.path.join(net_dir, "*.net"))):
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        file_stem = os.path.splitext(os.path.basename(path))[0].upper()
        for line in text.splitlines():
            if not line.lstrip().upper().startswith("X"):
                continue
            name = _subckt_name_from_x_line(line.strip())
            if not name:
                continue
            name = name.upper()
            paths = index.setdefault(name, [])
            if path not in paths:
                # File named after the part -> canonical template, goes first.
                if name == file_stem:
                    paths.insert(0, path)
                else:
                    paths.append(path)
    return index


def find_template(ic_name: str, index: dict[str, list[str]] | None = None,
                  net_dir: str = DEFAULT_NET_DIR) -> str | None:
    """Path of the best template netlist for `ic_name`, or None."""
    if index is None:
        index = build_template_index(net_dir)
    paths = index.get(ic_name.upper())
    return paths[0] if paths else None


if __name__ == "__main__":
    import sys

    idx = build_template_index()
    if len(sys.argv) > 1:
        ic = sys.argv[1].upper()
        print(f"{ic}: {idx.get(ic, '<< no template found >>')}")
    else:
        print(f"Indexed {len(idx)} distinct subcircuit names "
              f"from {DEFAULT_NET_DIR}")
        multi = {k: v for k, v in idx.items() if len(v) > 1}
        print(f"{len(multi)} parts appear in more than one netlist")
