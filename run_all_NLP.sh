#!/bin/bash
#
# run_all_NLP4OPT3.sh
#
# Purpose:
#   Convenience wrapper to run `run_NL4OPT.py` for a sweep of indices.
#   Behavior preserved: loops from 0..250 and prints the same echo line.
#
# Usage:
#   bash run_all_NLP4OPT3.sh
#
# Notes:
#   - Expects `python3` in PATH and `run_NL4OPT.py` in the same directory (or in PYTHONPATH).
#   - The variables "$opt_type" and "$difficulty" are intentionally *not* set here
#     to preserve the original echo behavior (they will print empty if unset).

# Loop through all combinations (0..250 inclusive)
for index in {0..250}; do
    echo "Running experiment with $opt_type, $difficulty, index $index"
    python3 run_NL4OPT.py --index "$index"
done
