# Data layout

- `templates/` contains the curated LTspice template corpus used by `template_index.py`.
- `verified_pairs.jsonl` contains simulator-verified adaptation records.

Only `.net` template sources are included. Generated LTspice `.raw`, `.log`, `.op.raw`, and editor `.cir` outputs are deliberately excluded.

Before publishing or redistributing this repository, verify the license and redistribution terms for every template netlist and vendor model referenced by the corpus.
