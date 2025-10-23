"""
run_NL4OPT.py

Purpose
-------
Entry point for running the NL4OPT experiment for a single problem index.
This version preserves behavior while adding documentation and structure.

Behavior preserved
------------------
- Same CLI (required: --index).
- Same default paths and filenames.
- Same model selection ('GPT4o-mini-1').
- Same dataframe reading (JSON lines) and column selection.
- Still runs at module import time (kept intentionally to preserve behavior
  for any existing tooling that imports this file and expects it to execute).

Notes
-----
- The code is grouped into small functions for readability; the main call is
  still made at module level to avoid changing the observable behavior.
"""

import argparse
import copy  # Preserved import even if not used downstream
import pandas as pd
from MCTS_used import MCTS

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments (behavior preserved)."""
    parser = argparse.ArgumentParser(description="Run MCTS optimization experiments.")
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="The problem index (between 0 and 20).",
    )
    # Original commented argument preserved for reference:
    # parser.add_argument('--gpt', type=str, choices=['GPT4o', 'GPT4o-mini-1'], required=True, help="The optimization type.")
    return parser.parse_args()


def _load_problem_dataframe(this_dir: str) -> pd.DataFrame:
    """
    Load the problem dataframe (JSON lines).

    Parameters
    ----------
    this_dir : str
        Base name used to construct the path 'data/{this_dir}.json'.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_json(f"data/{this_dir}.json", lines=True)


def _extract_problem(df: pd.DataFrame, index: int):
    """
    Extract the problem description and ground-truth fields given a row index.

    Returns
    -------
    tuple[str, Any, Any]
        (problem_str, ground_truth, problem_idx)
    """
    problem_str = df.iloc[index].en_question
    ground_truth = df.iloc[index].en_answer
    problem_idx = df.iloc[index].name
    return problem_str, ground_truth, problem_idx


def _run_single_experiment(index: int) -> None:
    """
    Run a single NL4OPT experiment using MCTS, preserving original behavior.

    Parameters
    ----------
    index : int
        Problem row index to run.
    """
    # Preserved path names
    this_dir = "NL4OPT"
    results_dir = "NL4OPT_results"

    # Load data and extract a single problem
    df = _load_problem_dataframe(this_dir)
    problem_str, ground_truth, problem_idx = _extract_problem(df, index)

    # Construct output directory for this problem and run MCTS
    this_mcts = MCTS()
    this_mcts.dfs_from_scratch(
        problem_str,
        f"{results_dir}/problem_{problem_idx}",
        ground_truth=ground_truth,
    )


# --- Preserve original "import side-effect" execution semantics ---
# The original module executed immediately upon import (argparse + run).
# To keep behavior identical, we still trigger execution at import time.
_args = _parse_args()
_run_single_experiment(_args.index)
