"""
MCTS_used.py

Purpose
-------
Implements:
- A Reasoning-oriented MCTSNode (ReasoningMCTSNode)
- A Monte-Carlo Tree Search driver (MCTS)
- Reward and RewardManager helpers
- Utility functions used within the MCTS process

This rewrite preserves original behavior and output while:
- Cleaning imports and removing exact duplicates
- Adding docstrings and comments for clarity
- Keeping names, file outputs, prints, constants, and flow the same
"""

from __future__ import annotations

# --- Standard library imports ---
import math
import random
import warnings
import os
import subprocess
import re
import json
import pickle
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Optional

# --- Third-party imports (behavior preserved) ---
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- Project imports (wildcards preserved intentionally) ---
from prompts import *
from utils import chat_gpt
from utils_prompt import *
from utils_exec import create_code_str, execute_python_code_vanilla, check_script_output
from utils_diff import filter_functionally_equivalent, filter_parameters
from utils_metrics import (
    compare_solutions_form,
    compare_solutions_form_score,
    obtain_prompt_solution_score_sol,
    obtain_prompt_solution_score_sol_partial,
)

# =========================
# Constants (unchanged)
# =========================
DEPTH_TERMINAL = 5
DEPTH_BEFORE_TERMINAL = 5
N_USED_GPT = 8 #N_USED_GPT = 10
N_DV_UNIQUE_GPT = 8 #N_DV_UNIQUE_GPT = 10
N_TOP = 3
N_REWARD = 5

# Steps / prompts mapping (unchanged)
DICT_PROMPT = {
    1: INST_PARAMETERS,
    2: INST_DV,
    3: INST_OBJ,
    4: INST_EQ_CONST,
    5: INST_INEQ_CONST,
}

DICT_STR = {
    1: "parameters",
    2: "decision_variables",
    3: "objective",
    4: "equality_constraints",
    5: "inequality_constraints",
}

# Execution config (unchanged)
THIS_DIRECTORY = "./results"
EXECUTABLE_STEPS = ["objective", "equality_constraints", "inequality_constraints"]
VARIABLE_STEPS = [
    "decision_variables",
    "objective",
    "equality_constraints",
    "inequality_constraints",
]
DV_SAME_PROMPT = False
GREEDY_PARAMETERS = True


# =========================
# Utilities
# =========================
def convert_to_native(obj):
    """
    Convert numpy scalars/arrays to Python native types for JSON serialization.

    Parameters
    ----------
    obj : Any

    Returns
    -------
    Any
        JSON-serializable native type.

    Raises
    ------
    TypeError
        If type is not supported.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Type {type(obj)} is not JSON serializable")


def parse_dict_with_comments(constraints_dict, comments_list):
    """
    Build a dict-string where each entry is optionally preceded by a comment.
    Behavior preserved (formatting and trailing commas are kept the same).
    """
    result = "{\n"
    for (key, value), comment in zip(constraints_dict.items(), comments_list):
        if comment:
            result += f"    # {comment}\n"
        result += f"    '{key}': '{value}',\n\n"
    result = result.rstrip(",\n\n") + "\n}"
    return result


def extract_comment(constraints_str, key):
    """
    Extract inline comment for a given key from an annotated dict-string.
    """
    pattern = r"#(.+)\n\s*'{}':".format(re.escape(key))
    match = re.search(pattern, constraints_str)
    if match:
        return match.group(1).strip()
    else:
        return None


def create_dict_string(this_dict_eval, this_dict_str):
    """
    Re-compose a dict-string preserving comments found in the original text.
    """
    all_comments = []
    for key, _ in this_dict_eval.items():
        all_comments += [extract_comment(this_dict_str, key)]
    output_string = parse_dict_with_comments(this_dict_eval, all_comments)
    return output_string


# --- DV checks: behavior preserved (including generous returns) ---
def check_single_decision_variable_set(var_set):
    """Verify each DV description contains a '<type>' placeholder; else False."""
    pattern = re.compile(r"<[^>]+>")
    try:
        for _, var_desc in var_set.items():
            if not pattern.search(var_desc):
                return False
    except Exception:
        return False
    return True


def is_decision_variable_name_correct(decision_variables):
    """
    Ensure base names (without bracketed indices) are unique.
    """
    base_names = set()
    for key in decision_variables.keys():
        base_name = re.sub(r"\[.*?\]", "", key).strip()
        if base_name in base_names:
            return False
        base_names.add(base_name)
    return True


def check_correctness_general_dv(decision_variables):
    """
    Ensure bracket count in keys/values is consistent (behavior preserved).
    """
    for key, description in decision_variables.items():
        indices_in_key = re.findall(r"\[([^\]]+)\]", key)
        num_indices_in_key = len(re.split(r"\s*,\s*", indices_in_key[0])) if indices_in_key else 0
        indices_in_description = re.findall(r"\{[^\}]*\}", description)
        num_indices_in_description = len(indices_in_description)
        if num_indices_in_key == 0 and num_indices_in_description > 0:
            return False
        elif num_indices_in_key != num_indices_in_description:
            return False
    return True


def is_well_parsed_decision_variables(decision_variables):
    """
    If key uses '[]', value should contain '{...}' somewhere (behavior preserved).
    """
    try:
        for key, value in decision_variables.items():
            if "[" in key and "]" in key:
                if not re.search(r"\{.*?\}", value):
                    return False
    except Exception:
        return False
    return True


def check_decision_variables_correctness_correct_brackets(decision_variables):
    """
    Reject DV descriptions that contain '{... in ...}' phrases (behavior preserved).
    """
    incorrect_variables = []
    pattern = re.compile(r"\{.*\bin\b.*\}")
    for var_name, description in decision_variables.items():
        if pattern.search(description):
            incorrect_variables.append(var_name)
    if incorrect_variables:
        return False
    return True


def check_decision_variables_correctness(dv_set):
    """
    Original code returns True unconditionally (preserved).
    """
    return True
    # is_type = check_single_decision_variable_set(dv_set)
    # is_right_indexing = is_well_parsed_decision_variables(dv_set)
    # correct_bracket_content = check_decision_variables_correctness_correct_brackets(dv_set)
    # general_correctness = check_correctness_general_dv(dv_set)
    # correct_name = is_decision_variable_name_correct(dv_set)
    # return is_type and is_right_indexing and correct_bracket_content and general_correctness and correct_name


def check_constraints_for_inequalities(constraints_dict):
    """
    Ensure no inequality operators exist within equality constraints string dict.
    """
    inequality_operators = ["<=", ">=", "<", ">"]
    for _, constraint_expression in constraints_dict.items():
        if any(op in constraint_expression for op in inequality_operators):
            return False
    return True


def check_constraints_for_correctness_obj(objective_dict):
    """
    Ensure objective has a single entry and does not contain '#'.
    """
    wrong_operators = ["#"]
    if len(objective_dict) != 1:
        return False
    this_value = list(objective_dict.values())[0]
    for op in wrong_operators:
        if op in this_value:
            return False
    return True


def modify_directory(directory, additional_string):
    """
    Add a prefix to the last path segment (behavior preserved).
    """
    head, tail = os.path.split(directory)
    new_tail = f"{additional_string}-{tail}"
    new_directory = os.path.join(head, new_tail)
    return new_directory


def check_none_key_value(d):
    """
    Return True iff dict contains key None and its value is also None.
    """
    return d.get(None) is None and None in d


def create_directories(path):
    """Create directories as needed (behavior preserved with print)."""
    os.makedirs(path, exist_ok=True)
    print(f"Directories created or already exist: {path}")


# =========================
# Pairwise ranking helpers
# =========================
def compute_ranking(pairwise_comparisons):
    """
    Compute Bradley–Terry ranking using logistic regression.

    Parameters
    ----------
    pairwise_comparisons : dict
        {i: {j: p_ij, ...}, ...} with p_ij in [0,1]

    Returns
    -------
    tuple[list[int], np.ndarray]
        (ranking list (best-first), abilities)
    """
    data = []
    labels = []

    solutions = sorted(
        set(pairwise_comparisons.keys())
        | set(j for sub in pairwise_comparisons.values() for j in sub)
    )

    for i in solutions:
        for j in solutions:
            if i < j:
                score_ij = pairwise_comparisons.get(i, {}).get(j, None)
                score_ji = pairwise_comparisons.get(j, {}).get(i, None)

                if score_ij is not None:
                    if not (0 <= score_ij <= 1):
                        raise ValueError(
                            f"Invalid score {score_ij} for comparison ({i}, {j}). Scores must be between 0 and 1."
                        )
                    if score_ij > 0.5:
                        data.append([i, j])
                        labels.append(1)
                    elif score_ij < 0.5:
                        data.append([i, j])
                        labels.append(0)
                elif score_ji is not None:
                    if not (0 <= score_ji <= 1):
                        raise ValueError(
                            f"Invalid score {score_ji} for comparison ({j}, {i}). Scores must be between 0 and 1."
                        )
                    if score_ji > 0.5:
                        data.append([i, j])
                        labels.append(0)
                    elif score_ji < 0.5:
                        data.append([i, j])
                        labels.append(1)

    data = np.array(data)
    labels = np.array(labels)

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Insufficient variation in labels. The model requires both wins and losses.")

    n_solutions = max(solutions) + 1
    X = np.zeros((len(data), n_solutions))
    for idx, (i, j) in enumerate(data):
        X[idx, i] = 1
        X[idx, j] = -1

    model = LogisticRegression(fit_intercept=False, solver="lbfgs")
    model.fit(X, labels)

    abilities = model.coef_[0]
    ranking = [solution for solution in np.argsort(-abilities) if solution in solutions]

    print("Final Ranking:")
    for rank, solution in enumerate(ranking, 1):
        print(f"Rank {rank}: Solution {solution} with ability score {abilities[solution]:.2f}")

    return ranking, abilities


# =========================
# Abstract Node
# =========================
class MCTSNode(ABC):
    """Abstract base class for MCTS nodes."""

    def find_children(self):
        self.children = self.children or self._get_children()
        return self.children

    @abstractmethod
    def find_one_child(self) -> "MCTSNode":
        raise NotImplementedError

    @property
    @abstractmethod
    def is_terminal(self):
        return True

    @property
    @abstractmethod
    def reward(self):
        return 0

    @property
    @abstractmethod
    def visited(self):
        return 0


# =========================
# Reasoning MCTS Node
# =========================
class ReasoningMCTSNode(MCTSNode):
    """
    MCTS node specialized for structured reasoning over problem formulation steps.
    Behavior (file I/O, prints, ranking logic) preserved as-is.
    """

    @property
    def visited(self):
        return self._visited

    def __init__(
        self,
        problem_str,
        path_used,
        depth=0,
        form_dict_str={},
        form_dict_eval={},
        parent: "ReasoningMCTSNode" = None,
        engine_used="GPT4o",
        idx=1,
        verbose_prompt=True,
        verbose_solution=True,
        n_reward=N_REWARD,
        n_used_gpt=N_USED_GPT,
        n_top=N_TOP,
        specific_n_dict={},
    ):
        # Core
        self._conf = None
        self.problem_str = problem_str
        self.prompt_problem = TEMPLATE_PROBLEM.replace("###PROBLEM DESCRIPTION###", problem_str)
        self.engine_used = engine_used
        self.children = []
        self.depth = depth
        self._visited = False
        self.parent = parent
        self.this_dir = path_used
        self.this_step = DICT_STR[depth] if depth > 0 else None
        self.next_step = DICT_STR[depth + 1] if depth < DEPTH_TERMINAL else None
        self.idx = idx
        self.verbose_prompt = verbose_prompt
        self.verbose_solution = verbose_solution
        self.specific_n_dict = specific_n_dict
        create_directories(self.this_dir)

        self.form_dict_str = form_dict_str
        self.form_dict_eval = form_dict_eval

        # Preserve original equality constraints special-case file writes
        if self.this_step == "equality_constraints":
            self.save_json(self.form_dict_str, "A-form-eval-string-original")
            if not check_none_key_value(self.form_dict_eval[self.this_step]):
                self.form_dict_str[self.this_step] = create_dict_string(
                    form_dict_eval[self.this_step], form_dict_str[self.this_step]
                )

        if depth > 0:
            self.save_json(self.form_dict_str, "A-form-eval-string")

        # GPT gen control
        self.n_reward = n_reward
        self.n_top = n_top
        self.n_used_gpt = (
            n_used_gpt if not self.this_step in specific_n_dict else specific_n_dict[self.this_step]
        )

        # Node status flags
        self.is_node_correct = True
        self.run_code_if_code = True

        if self.this_step == "equality_constraints" and check_none_key_value(
            self.form_dict_eval[self.this_step]
        ):
            self.run_code_if_code = False

        if depth > 0:
            if len(form_dict_eval[self.this_step]) > 0:
                self.not_empty = True
                if self.run_code_if_code:
                    if self.this_step == "objective":
                        self.not_empty = check_constraints_for_correctness_obj(
                            self.form_dict_eval[self.this_step]
                        )
                        if not self.not_empty:
                            self.run_code_if_code = False
                    if self.this_step == "decision_variables":
                        self.not_empty = check_decision_variables_correctness(
                            self.form_dict_eval[self.this_step]
                        )
            else:
                self.not_empty = False
                self.run_code_if_code = False

        if self.this_step == "inequality_constraints":
            self.reward_backpropagation = Reward()
        else:
            self.reward_backpropagation = []

        # Executable steps (produce code and run)
        if self.depth > 0 and self.this_step in EXECUTABLE_STEPS and self.run_code_if_code:
            try:
                code_str = create_code_str(self.form_dict_eval, variables_included=self.this_step)
            except Exception as e:
                self.is_node_correct = False
                self.save_prompt("Could not create code: " + str(e), "A-output_runnable")
            if self.is_node_correct:
                self.output_str = execute_python_code_vanilla(
                    code_str,
                    python_filename=f"{self.this_dir}/A-python_runnable.py",
                    output_filename=f"{self.this_dir}/A-output_runnable.txt",
                )
                self.is_node_correct, _ = check_script_output(f"{self.this_dir}/A-output_runnable.txt")

                if self.this_step == "inequality_constraints" and check_none_key_value(
                    self.form_dict_eval[self.this_step]
                ):
                    self.run_code_if_code = False

        if self.depth > 0:
            try:
                self.save_json(form_dict_eval, "A-form_eval", transform_json=True)
            except Exception:
                self.not_empty = False

    def get_count_node(self):
        if self.this_step == "inequality_constraints":
            return self.reward_backpropagation.count
        else:
            if len(self.reward_backpropagation) == 0:
                return 0
            else:
                count = 0
                for rwd in self.reward_backpropagation:
                    count += rwd.count
                return count

    def get_uct_reward(self):
        reward = 0
        for rwd in self.reward_backpropagation:
            reward += rwd.reward
        return reward

    def update_reward(self, r0):
        self.r0 = r0

    # Intentionally keep this odd, in-class import to preserve original structure
    from utils_prompt import convert_tuple_keys_to_str, convert_str_keys_to_tuple  # noqa: E402

    def save_json(self, this_dict, json_name, transform_json=False):
        if transform_json:
            this_dict = convert_tuple_keys_to_str(this_dict)
        with open(f"{self.this_dir}/{json_name}.json", "w") as file:
            json.dump(this_dict, file, indent=4)

    def save_prompt(self, prompt_to_gpt, name_prompt):
        with open(f"{self.this_dir}/{name_prompt}.txt", "w") as file:
            file.write(prompt_to_gpt)

    def load_python_script(self):
        with open(f"{self.this_dir}/python_runnable.py", "r") as file:
            script_content = file.read()
        return script_content

    def create_prompt_gpt(self):
        self.prompt_to_gpt = obtain_new_prompt(
            self.prompt_problem, self.form_dict_str, DICT_PROMPT[self.depth + 1]
        )
        self.save_prompt(self.prompt_to_gpt, "prompt_child_gpt")
        return self.prompt_to_gpt

    def _child_node(self, form_str, form_eval, idx):
        return ReasoningMCTSNode(
            self.problem_str,
            f"{self.this_dir}/{idx}-{self.next_step}",
            self.depth + 1,
            form_str,
            form_eval,
            parent=self,
            idx=idx,
            engine_used=self.engine_used,
            n_reward=self.n_reward,
            n_top=self.n_top,
            n_used_gpt=self.n_used_gpt,
            specific_n_dict=self.specific_n_dict,
        )

    def get_gpt_response(self):
        prompt_to_gpt = self.create_prompt_gpt()
        response = chat_gpt(user_prompt=prompt_to_gpt, n_used=self.n_used_gpt, engine_used=self.engine_used)
        n_gen = self.n_used_gpt

        all_form_str = []
        all_form_eval = []
        for idx in range(n_gen):
            form_dict_str_cp, form_dict_eval_cp = self.form_dict_str.copy(), self.form_dict_eval.copy()
            content = response.choices[idx].message.content
            try:
                result_string_par, results_dict_par = extract_dict_from_string(content, self.next_step)
            except Exception:
                continue
            form_dict_str_cp[DICT_STR[self.depth + 1]] = result_string_par
            form_dict_eval_cp[DICT_STR[self.depth + 1]] = results_dict_par
            all_form_str += [form_dict_str_cp]
            all_form_eval += [form_dict_eval_cp]
        return all_form_str, all_form_eval

    def _get_solution_prompt(self, possible_children):
        for idx, this_child in enumerate(possible_children):
            if idx == 0:
                solution_prompt = "\\ \\n["
            solution_prompt += f"{self.next_step}_{idx}: {this_child.form_dict_str[self.next_step]}\n"
        solution_prompt += "]"
        return solution_prompt

    def _get_gpt_unique_dv(self, correct_children):
        solution_prompt = self._get_solution_prompt(correct_children)
        prompt_to_gpt = get_rank_filtering_prompt(
            INST_GROUP_DV, self.problem_str, self.form_dict_str, solution_prompt
        )
        if self.verbose_prompt:
            self.save_prompt(prompt_to_gpt, "prompt_unique_dv")
        response = chat_gpt(user_prompt=prompt_to_gpt, n_used=N_DV_UNIQUE_GPT, engine_used=self.engine_used)
        list_content = [response.choices[i].message.content for i in range(N_DV_UNIQUE_GPT)]
        if self.verbose_prompt:
            self.save_prompt(list_content[0], "resp_unique_dv")
        indx_groups = extract_unique_group_idx(list_content, len(correct_children), self.next_step)
        return [correct_children[idx] for idx in indx_groups]

    def _get_possible_children(self):
        all_form_str, all_form_eval = self.get_gpt_response()
        possible_children = []
        idx = 0
        for this_form_str, this_form_eval in zip(all_form_str, all_form_eval):
            possible_children += [self._child_node(this_form_str, this_form_eval, idx)]
            idx += 1
        return possible_children, all_form_str, all_form_eval

    def _get_not_empty_children(self, possible_children):
        no_empty_children = []
        empty_children = []
        for this_node in possible_children:
            if this_node.not_empty:
                no_empty_children += [this_node]
            else:
                empty_children += [this_node]
        return no_empty_children, empty_children

    def _get_correct_children(self, possible_children):
        correct_children = []
        incorrect_children = []
        for this_node in possible_children:
            if this_node.is_node_correct:
                correct_children += [this_node]
            else:
                incorrect_children += [this_node]
        return correct_children, incorrect_children

    def _get_unique_children(self, correct_children):
        if len(correct_children) > 1:
            if self.next_step != "decision_variables":
                aux_path = f"{self.this_dir}/aux"
                if self.next_step != "parameters":
                    create_directories(aux_path)
                else:
                    if GREEDY_PARAMETERS:
                        unique_children, discarded_children = filter_parameters(correct_children)
                        return unique_children, discarded_children
                unique_children, discarded_children = filter_functionally_equivalent(correct_children, aux_path)
            else:
                if not DV_SAME_PROMPT:
                    unique_children = self._get_gpt_unique_dv(correct_children)
                else:
                    unique_children = correct_children
                discarded_children = [obj for obj in correct_children if not (obj in unique_children)]
        else:
            unique_children, discarded_children = correct_children, []
        return unique_children, discarded_children

    def _get_children_reward(self, unique_objects):
        if len(unique_objects) > 1:
            solution_prompt = self._get_solution_prompt(unique_objects)
            prompt_to_gpt = get_rank_filtering_prompt(
                INST_RANK.replace("#VARIABLE#", self.next_step),
                self.problem_str,
                self.form_dict_str,
                solution_prompt,
            )
            if self.verbose_prompt:
                self.save_prompt(prompt_to_gpt, "prompt_reward")
            response = chat_gpt(user_prompt=prompt_to_gpt, n_used=self.n_reward, engine_used=self.engine_used)
            current_dict = {f"{self.next_step}_{idx}": 0 for idx in range(len(unique_objects))}
            for i in range(self.n_reward):
                content = response.choices[i].message.content
                if self.verbose_prompt and i == 0:
                    self.save_prompt(prompt_to_gpt, "response_reward_0")
                try:
                    dict_rank = extract_dictionary_rank(content, len(unique_objects), self.next_step)
                    if dict_rank is not None:
                        current_dict = update_rank(current_dict, dict_rank)
                except Exception:
                    print("A")
            return get_normalized_rank(current_dict), current_dict
        else:
            return np.array([1.0]), None

    def _get_children(self, n_used_gpt=None, n_top=None):
        if n_used_gpt is not None:
            self.n_used_gpt = n_used_gpt
        if n_top is not None:
            self.n_top = n_top
        self._visited = True

        if self.is_terminal:
            return self.children

        possible_children, _, _ = self._get_possible_children()
        self.save_results_children(possible_children, "children_possible")
        self.save_children_outputs(possible_children)

        no_empty_children, empty_children = self._get_not_empty_children(possible_children)
        self.rename_children(empty_children, "z-empty")

        correct_children, incorrect_children = self._get_correct_children(no_empty_children)
        self.save_results_children(incorrect_children, "output_incorrect_children")
        self.rename_children(incorrect_children, "z-incorrect")

        unique_children, discarded_children = self._get_unique_children(correct_children)
        self.save_results_children(unique_children, "children_unique")
        self.rename_children(discarded_children, "z-clustered")

        all_rewards, _ = self._get_children_reward(unique_children)
        self.save_rewards(unique_children, all_rewards)

        lowest_reward_children = []
        highest_rewards = np.sort(all_rewards)[::-1][: self.n_top]
        for this_node, this_reward in zip(unique_children, all_rewards):
            this_node.update_reward(this_reward)
            if this_reward in highest_rewards:
                self.children += [this_node]
            else:
                lowest_reward_children += [this_node]
        self.rename_children(lowest_reward_children, "low-scored")
        return self.children

    def save_children_outputs(self, possible_children):
        if self.depth > 0 and self.this_step in EXECUTABLE_STEPS:
            all_text = ""
            for this_node in possible_children:
                if this_node.run_code_if_code:
                    with open(f"{this_node.this_dir}/A-output_runnable.txt", "r") as file:
                        output_content = file.read()
                all_text += f"------------------------------------------------ {this_node.this_step}_{this_node.idx}, "
                if this_node.run_code_if_code:
                    all_text += f"is correct: {this_node.is_node_correct} ------------------------------------------------"
                    all_text += f"\n\n{output_content}\n\n"
                else:
                    all_text += (
                        "No constraint -------------------------------------------------------------------------"
                    )
                all_text += "\n\n"
            self.save_prompt(all_text, "output_code_correct")

    def save_results_children(self, these_children, name_file):
        all_dict = {}
        for this_child in these_children:
            all_dict[f"{this_child.this_step}_{this_child.idx}"] = this_child.form_dict_eval[
                this_child.this_step
            ]
        self.save_json(all_dict, name_file, transform_json=True)

    def save_rewards(self, unique_children, all_rewards):
        all_dict = {}
        for i, child in enumerate(unique_children):
            all_dict[child.idx] = all_rewards[i]
        self.save_json(all_dict, "children_rewards_idx")

    def rename_children(self, these_children, add_name):
        for child in these_children:
            os.rename(child.this_dir, modify_directory(child.this_dir, add_name))

    def find_children(self, n_used_this=None):
        self.children = self.children or self._get_children()
        return self.children

    def find_one_child(self) -> MCTSNode:
        return random.choice(self.find_children())

    def _calculate_reward(self):
        self.prompt, self._r1, self._ans_list = self.reward_fn(self.prompt, self.depth)

    def compute_final_reward(self):
        return None

    @property
    def is_before_terminal(self):
        return self.depth >= DEPTH_BEFORE_TERMINAL

    @property
    def is_before_terminal_real(self):
        return self.depth >= DEPTH_BEFORE_TERMINAL - 1

    @property
    def is_terminal(self):
        return self.depth >= DEPTH_TERMINAL

    @property
    def reward(self):
        return self._r0

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.gen_fn is None or self.reward_fn is None:
            warnings.warn("MCTSNode from pickle is read-only; Do not further roll out the tree!")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["gen_fn"] = None
        state["reward_fn"] = None
        return state


# =========================
# Dict/path helpers
# =========================
def create_nested_dict(keys):
    nested_dict = current = {}
    for key in keys:
        current[key] = {}
        current = current[key]
    return nested_dict, current


def extract_results(d):
    result_list = []

    def recursive_extract(d):
        for _, value in d.items():
            if isinstance(value, dict):
                if all(not isinstance(v, dict) for v in value.values()):
                    result_list.append(value)
                else:
                    recursive_extract(value)

    recursive_extract(d)
    return result_list


def update_nested_dict(dict1, dict2):
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1 and isinstance(dict1[key], dict):
            update_nested_dict(dict1[key], value)
        else:
            dict1[key] = value
    return dict1


# =========================
# MCTS driver
# =========================
class MCTS:
    """
    Monte-Carlo Tree Search driver coordinating node expansion, ranking,
    and result collection. Original prints, file paths, and behavior preserved.
    """

    def __init__(
        self,
        engine_used="GPT4o",
        comparison_type="rank",
        alpha=1.0,
        w_exp=1,
        discount=1,
        prior=False,
        aggr_reward="sum",
        aggr_child="max",
        n_reward=N_REWARD,
        n_used_gpt=N_USED_GPT,
        n_top=N_TOP,
        specific_n_dict={},
        lambda_cte=0.5,
        num_warmup=3,
    ):
        self.Q = {}
        self.N = {}
        self.fR = {}
        self.all_nodes_comparisons = {}
        self.all_nodes_scores = {}
        self.all_nodes_order = {}
        self.all_visited_nodes = []
        self.dumb_nodes = []
        self.all_results = {}
        self.r_baseline = {}

        self.terminal_nodes = []
        self.children = dict()

        self.alpha = alpha
        self.w_exp = w_exp
        self.discount = discount
        self.prior = prior
        self.aggr_reward = aggr_reward
        self.aggr_child = aggr_child
        self.engine_used = engine_used
        self.comparison_type = comparison_type
        self.n_reward = n_reward
        self.n_used_gpt = n_used_gpt
        self.n_top = n_top
        self.specific_n_dict = specific_n_dict
        self.collected_results_path = {}
        self.ground_truth = None
        self.baselines_results = []

        self.lambda_cte = lambda_cte
        self.num_warmup = num_warmup

    def update_collected_data(self, path, collected_rewards, collected_path, result_str):
        new_nested_dict, new_dict = create_nested_dict(collected_path)
        new_dict["accumulated_reward"] = sum(collected_rewards)
        new_dict.update(parse_gurobi_output(result_str))
        if self.ground_truth is not None:
            new_dict["ground_truth"] = self.ground_truth

        update_nested_dict(self.collected_results_path, new_nested_dict.copy())
        collected_all_results = extract_results(self.collected_results_path)

        with open(f"{path}/all_results_path.json", "w") as file:
            json.dump(self.collected_results_path, file, indent=4)
        with open(f"{path}/all_results.jsonl", "w") as json_file:
            for dictionary in collected_all_results:
                json_file.write(json.dumps(dictionary) + "\n")

    def update_prior_reward(self, node, prior_reward_list):
        if node.this_step in VARIABLE_STEPS:
            n_prior_reward_list = [reward for reward in prior_reward_list]
            n_prior_reward_list += [node.r0]
            return n_prior_reward_list
        else:
            return []

    def update_path_list(self, node, prior_path_list):
        if node.this_step in VARIABLE_STEPS:
            n_prior_path_list = [this_path for this_path in prior_path_list]
            n_prior_path_list += [f"{node.this_step} (idx: {node.idx}) (R: {node.r0:.2f})"]
            return n_prior_path_list
        else:
            return []

    def save_node(self, node, path):
        with open(f"{path}/initial_dummy_node.pkl", "wb") as file:
            pickle.dump(node, file)

    def dfs_from_scratch(self, problem_description, path, ground_truth=None):
        self.path = path
        self.ground_truth = ground_truth
        dummy_node = ReasoningMCTSNode(
            problem_description,
            path,
            n_reward=self.n_reward,
            engine_used=self.engine_used,
            n_top=self.n_top,
            n_used_gpt=self.n_used_gpt,
            specific_n_dict=self.specific_n_dict,
        )
        collected_prior_reward = []
        collected_path = []
        self.dfs(dummy_node, collected_path=collected_path, collected_prior_reward=collected_prior_reward)
        self.save_node(dummy_node, path)

    def dfs(self, node, collected_path=[], collected_prior_reward=[]):
        collected_path = self.update_path_list(node, collected_path)
        collected_prior_reward = self.update_prior_reward(node, collected_prior_reward)
        self._expand(node)
        print("depth, ", node.this_step)
        if node.this_step == "inequality_constraints":
            self.update_collected_data(self.path, collected_prior_reward.copy(), collected_path.copy(), node.output_str)
            return

        for child in node.children:
            self.dfs(child, collected_path.copy(), collected_prior_reward.copy())

    def one_rollout(self, problem_description, path):
        self.dummy_node = ReasoningMCTSNode(
            problem_description,
            path,
            n_reward=self.n_reward,
            engine_used=self.engine_used,
            n_top=self.n_top,
            n_used_gpt=self.n_used_gpt,
            specific_n_dict=self.specific_n_dict,
        )
        path = self._select_prior(self.dummy_node)
        return path

    def save_this_object(self, path, idx, add_name=None):
        if add_name is None:
            path_to_save = f"{path}/MCTS_object_{idx}.pkl"
        else:
            path_to_save = f"{path}/MCTS_object_{add_name}_{idx}.pkl"
        with open(path_to_save, "wb") as file:
            pickle.dump(self, file)

    def get_pairwise_comparison(self):
        new_dict = {}
        for this_node in self.all_nodes_order:
            if this_node not in self.dumb_nodes:
                aux_dict = {
                    self.all_nodes_order[key]: value
                    for key, value in self.all_nodes_comparisons[this_node].items()
                }
                new_dict[self.all_nodes_order[this_node]] = aux_dict
        return new_dict

    def get_ranking_scores(self):
        aux_dict = {self.all_nodes_order[this_node]: self.all_nodes_scores[this_node] for this_node in self.all_nodes_order}
        aux_order = np.array(list(aux_dict.keys()))
        aux_scores = np.array(list(aux_dict.values()))
        arg_scores = aux_scores.argsort()[::-1]
        return aux_order[arg_scores]

    def get_ranking_from_pairwise_comparison(self):
        pairwise_comparisons = self.get_pairwise_comparison()
        if len(self.baselines_results) < 2:
            return None, None
        if len(self.baselines_results) == 2:
            ranking_results = [0, 1] if pairwise_comparisons[1][0] >= 0.5 else [1, 0]
        elif len(self.baselines_results) > 2:
            ranking_results, _ = compute_ranking(pairwise_comparisons)
        ranking_results = list(ranking_results)
        ranking_results += [self.all_nodes_order[this_dumb_node] for this_dumb_node in self.dumb_nodes]
        return pairwise_comparisons, ranking_results

    def update_collected_result_rollout(self, node, final_reward=None):
        to_update_dict = {}
        output_gurobipy = parse_gurobi_output(node.output_str)
        to_update_dict.update(output_gurobipy)
        if final_reward is not None:
            to_update_dict["final_reward"] = final_reward

        if self.global_scoring == "ranking":
            try:
                pairwise_comparisons, ranking_results = self.get_ranking_from_pairwise_comparison()
            except Exception:
                pairwise_comparisons, ranking_results = None, None
            to_update_dict["pairwise_comparison"] = pairwise_comparisons
            to_update_dict["baselines"] = [self.all_nodes_order[this_node] for this_node in self.baselines_results]

        elif self.global_scoring == "score":
            ranking_results = self.get_ranking_scores()

        aux_order_dict = {value: key for key, value in self.all_nodes_order.items()}
        to_update_dict["ranking_result"] = list(ranking_results) if ranking_results is not None else None
        to_update_dict["all_results"] = (
            [self.all_results[aux_order_dict[int(i_rank)]] for i_rank in to_update_dict["ranking_result"]]
            if ranking_results is not None
            else None
        )
        if self.global_scoring == "score":
            to_update_dict["all_scores"] = (
                [self.all_nodes_scores[aux_order_dict[int(i_rank)]] for i_rank in to_update_dict["ranking_result"]]
                if ranking_results is not None
                else None
            )

        self.all_results_best[self.count_many_rollout] = to_update_dict

        idx_used_to_save = 4
        if self.count_many_rollout < 4:
            idx_used_to_save = 4
        elif self.count_many_rollout < 8 and self.count_many_rollout >= 4:
            idx_used_to_save = 8
        elif self.count_many_rollout < 16 and self.count_many_rollout >= 8:
            idx_used_to_save = 16

        add_name = None
        if self.update_baseline:
            add_name = f"updateb_{self.local_scoring}_{self.global_scoring}"
        else:
            add_name = f"{self.local_scoring}_{self.global_scoring}"

        self.save_this_object(self.path_dummy_node, idx_used_to_save, add_name)
        if add_name is None:
            json_path = f"{self.path_dummy_node}/all_results_MCTS_{idx_used_to_save}.json"
        else:
            json_path = f"{self.path_dummy_node}/all_results_MCTS_{add_name}_{idx_used_to_save}.json"

        with open(json_path, "w") as file:
            json.dump(self.all_results_best, file, indent=4, default=convert_to_native)

        self.count_many_rollout += 1

    def load_dummy(self):
        with open(f"{self.path_dummy_node}/initial_dummy_node.pkl", "rb") as file:
            dummy_node = pickle.load(file)
        return dummy_node

    def update_global_scoring(self, global_scoring="ranking"):
        self.global_scoring = global_scoring

    def update_local_scoring(self, local_scoring=None):
        self.local_scoring = local_scoring

    def update_update_baseline(self, update_baseline=False):
        self.update_baseline = update_baseline

    def many_rollouts(self, path_dummy_node, total_rollouts, global_scoring="ranking", local_scoring=None, update_baseline=False):
        self.path_dummy_node = path_dummy_node
        dummy_node = self.load_dummy()
        self.update_global_scoring(global_scoring)
        self.update_local_scoring(local_scoring)
        self.update_update_baseline(update_baseline=update_baseline)

        self.all_results_best = {}
        self.count_many_rollout = 0

        num_rollout = 0
        max_dummy_rollouts = 0
        while True:
            if max_dummy_rollouts > 999:
                print("AAAAAAAAAAA")
                break
            if num_rollout == total_rollouts:
                break
            succesful_rollout = self.rollout(dummy_node, not_create_children=True)
            if succesful_rollout:
                num_rollout += 1
            max_dummy_rollouts += 1

    def update_order_node(self, node):
        if node not in self.all_nodes_order:
            self.all_nodes_order[node] = len(self.all_nodes_order)
        if self.global_scoring == "ranking":
            if node not in self.all_nodes_comparisons:
                self.all_nodes_comparisons[node] = {}

    def update_baselines_results(self):
        if len(self.baselines_results) >= self.num_warmup:
            pairwise_comparisons, ranking_results = self.get_ranking_from_pairwise_comparison()
            best_rankig_results = ranking_results[: self.num_warmup]
            self.baselines_results = [
                this_node for this_node in self.all_nodes_order if self.all_nodes_order[this_node] in best_rankig_results
            ]

    def rollout(self, node: MCTSNode, not_create_children=False):
        path = self._select_prior(node, not_create_children=not_create_children)

        if path[-1].this_step != "inequality_constraints":
            self._back_propagate(path, reward=0)
            return False

        self.update_order_node(path[-1])
        float_output_node, is_achieve_solution = self.check_and_get_best_objective(path[-1])

        if path[-1] not in self.all_results:
            self.all_results[path[-1]] = float_output_node
            if not is_achieve_solution:
                self.dumb_nodes += [path[-1]]

        effective = True
        if path[-1] in self.r_baseline:
            reward = self.r_baseline[path[-1]]
            effective = False
        else:
            reward = self.get_reward_update_ranking(path[-1]) if is_achieve_solution or self.global_scoring == "score" else 0
            if is_achieve_solution and (len(self.baselines_results) < self.num_warmup) and not (self.global_scoring == "score"):
                self.baselines_results += [path[-1]]
                reward = 0.5
            self.r_baseline[path[-1]] = reward
            self.update_collected_result_rollout(path[-1], final_reward=reward)
            if self.update_baseline:
                self.update_baselines_results()

        self._back_propagate(path, reward=reward)
        return effective

    def _select_prior(self, node: MCTSNode, not_create_children=False):
        path = [node]
        while not node.this_step == "inequality_constraints":
            print("this step: ", node.this_step, ", this idx: ", node.idx)
            try:
                self._expand(node, not_create_children=not_create_children)
            except Exception:
                import pdb  # preserved debug hook
                pdb.set_trace()
                print("a")
            if len(self.children[node]) == 0:
                return path
            node = self._uct_select(node)
            path.append(node)
            if node not in self.all_visited_nodes:
                self.all_visited_nodes += [node]
        print("---------------------")
        return path

    def _get_children_reward(self, parent_node, children_nodes):
        if self.local_scoring == "ranking":
            if len(children_nodes) > 1:
                solution_prompt = parent_node._get_solution_prompt(children_nodes)
                prompt_to_gpt = get_rank_filtering_prompt(
                    INST_RANK.replace("#VARIABLE#", parent_node.next_step),
                    parent_node.problem_str,
                    parent_node.form_dict_str,
                    solution_prompt,
                )
                response = chat_gpt(user_prompt=prompt_to_gpt, n_used=parent_node.n_reward, engine_used=self.engine_used)
                current_dict = {f"{parent_node.next_step}_{idx}": 0 for idx in range(len(children_nodes))}
                for i in range(parent_node.n_reward):
                    content = response.choices[i].message.content
                    try:
                        dict_rank = extract_dictionary_rank(content, len(children_nodes), parent_node.next_step)
                        if dict_rank is not None:
                            current_dict = update_rank(current_dict, dict_rank)
                    except Exception:
                        print("A")
                return get_normalized_rank(current_dict), current_dict
            else:
                return np.array([0.5])
        elif self.local_scoring == "score":
            all_rewards = []
            for child in children_nodes:
                all_rewards += [obtain_score_from_node_partial(child, self.engine_used, 2)]
            return all_rewards

    def check_if_update_local_reward(self):
        if not hasattr(self, "local_scoring"):
            return False
        else:
            if self.local_scoring is None:
                return False
            else:
                return True

    def _expand(self, node: MCTSNode, not_create_children=False, n_used_this=1):
        if node not in self.children:
            if not_create_children:
                children_node = node.children
            else:
                children_node = node.find_children()

            if self.check_if_update_local_reward() and node.this_step is not None:
                these_rewards = self._get_children_reward(node, children_node)
                for child, reward in zip(children_node, these_rewards):
                    child.update_reward(reward)

            self.children[node] = children_node

    def _back_propagate(self, path, reward):
        for node in reversed(path):
            if node not in self.N:
                self.N[node] = 0
            if node not in self.Q:
                self.Q[node] = 0

            self.N[node] += 1
            self.Q[node] += reward
            print("updating: ", node.this_step, " , idx", node.idx)

    def _uct(self, node: MCTSNode, log_n_f: float):
        if node not in self.N:
            self.N[node] = 0
        if self.N[node] == 0:
            return node.r0 + self.w_exp * math.sqrt(log_n_f)
        else:
            value = self.lambda_cte * node.r0 + (1 - self.lambda_cte) * self.Q[node] / self.N[node]
            return value + self.w_exp * math.sqrt(log_n_f / self.N[node])

    def _uct_select(self, node: MCTSNode, prior=True):
        if node not in self.N:
            self.N[node] = 0
        if prior and self.N[node] == 0:
            log_n = math.log(1)
        else:
            log_n = math.log(self.N[node])
        return max(self.children[node], key=lambda n: self._uct(n, log_n))

    def _greedy(self, node: MCTSNode):
        if not hasattr(self, "N2"):
            if node not in self.N:
                return node.r0
            else:
                value = self.lambda_cte * node.r0 + (1 - self.lambda_cte) * self.Q[node] / self.N[node]
                return value
        else:
            node_str = filter_path(node.this_dir)
            if node_str in self.N2 and node_str in self.Q2 and not (node_str in self.N2 and self.N2[node_str] == 0):
                value = self.lambda_cte * node.r0 + (1 - self.lambda_cte) * self.Q2[node_str] / self.N2[node_str]
                return value
            else:
                return node.r0

    def _greedy_select(self, node: MCTSNode):
        return max(self.children[node], key=lambda n: self._greedy(n))

    def greedy_search_ground_truth(self, ground_truth, path_to_dummy_node, node_name="initial_dummy_node"):
        with open(f"{path_to_dummy_node}/{node_name}.pkl", "rb") as file:
            dummy_node = pickle.load(file)

        path = []
        unique_leaf_values = set()
        leaf_values = []

        found = self._greedy_search_recursive(dummy_node, ground_truth, path, unique_leaf_values, leaf_values)

        if found:
            print("Ground truth found!")
            print(f"Number of unique leaf nodes visited: {len(unique_leaf_values)}")
            return path, len(unique_leaf_values), leaf_values
        else:
            print("Ground truth not found in the tree.")
            print(f"Number of unique leaf nodes visited: {len(unique_leaf_values)}")
            return None, len(unique_leaf_values), leaf_values

    def check_ground_truth(self, node_value, ground_truth):
        tolerance = 0.05
        try:
            ground_truth = float(ground_truth)
            node_value = float(node_value)
            if ground_truth != 0:
                if abs((node_value - ground_truth) / ground_truth) <= tolerance:
                    return True
                else:
                    return False
            else:
                if abs(node_value) <= tolerance:
                    return True
                else:
                    return False
        except Exception:
            return False

    def _greedy_search_recursive(self, node, ground_truth, path, unique_leaf_values, leaf_values):
        path.append(node)

        if node.this_step == "inequality_constraints":
            node_value = self.get_node_value(node)
            unique_leaf_values.add(node_value)
            leaf_values.append(node_value)
            if self.check_ground_truth(node_value, ground_truth):
                return True
            else:
                path.pop()
                return False

        self._expand(node, not_create_children=True)

        if node not in self.children or len(self.children[node]) == 0:
            path.pop()
            return False

        children = self.children[node]
        sorted_children = sorted(children, key=lambda n: self._greedy(n), reverse=True)

        for child in sorted_children:
            found = self._greedy_search_recursive(child, ground_truth, path, unique_leaf_values, leaf_values)
            if found:
                return True

        path.pop()
        return False

    def get_node_value(self, node):
        result = parse_gurobi_output(node.output_str)
        try:
            result_float = float(result["best_objective"])
        except Exception:
            result_float = result["best_objective"]
        return result_float

    def check_and_get_best_objective(self, node: ReasoningMCTSNode):
        try:
            node_output = parse_gurobi_output(node.output_str)
            best_objective = node_output["best_objective"]
            is_achieve_solution = True
            print("node, ", node_output)
        except Exception:
            float_output_node = -9999
            is_achieve_solution = False
        try:
            float_output_node = float(best_objective)
        except Exception:
            float_output_node = -9999
            is_achieve_solution = False
        return float_output_node, is_achieve_solution

    def get_reward_update_ranking(self, node: ReasoningMCTSNode):
        N_EVAL = 5
        if self.global_scoring == "ranking":
            if len(self.baselines_results) == 0:
                return None
            dict_node_new = self.all_nodes_comparisons[node]

            all_wins_s1, all_wins_s2 = 0, 0
            for node_compar in self.baselines_results:
                dict_node_compar = self.all_nodes_comparisons[node_compar]
                success = False
                while not success:
                    try:
                        i1_aux_wins_s1, i1_aux_wins_s2 = compare_solutions_form(
                            node.form_dict_str, node_compar.form_dict_str, N_EVAL, node.problem_str, self.engine_used
                        )
                        i2_aux_wins_s2, i2_aux_wins_s1 = compare_solutions_form(
                            node_compar.form_dict_str, node.form_dict_str, N_EVAL, node.problem_str, self.engine_used
                        )
                        aux_wins_s1 = i1_aux_wins_s1 + i2_aux_wins_s1
                        aux_wins_s2 = i1_aux_wins_s2 + i2_aux_wins_s2
                        success = True
                    except Exception as e:
                        print(f"An error occurred: {e}. Retrying...")
                        continue

                all_wins_s1 += aux_wins_s1
                all_wins_s2 += aux_wins_s2

                dict_node_new[node_compar] = aux_wins_s1 / (aux_wins_s1 + aux_wins_s2)
                dict_node_compar[node] = aux_wins_s2 / (aux_wins_s1 + aux_wins_s2)

            final_score = all_wins_s1 / (all_wins_s1 + all_wins_s2)
            return final_score

        elif self.global_scoring == "score":
            final_score = obtain_score_from_node(node, self.engine_used, N_EVAL)
            self.all_nodes_scores[node] = final_score
            return final_score

    def create_new_node_N_Q(self):
        self.N2 = {filter_path(key.this_dir): value for key, value in self.N.items()}
        self.Q2 = {filter_path(key.this_dir): value for key, value in self.Q.items()}


def obtain_score_from_node_partial(node, engine_used, n_eval):
    success = False
    while not success:
        try:
            prompt_to_gpt = obtain_prompt_solution_score_sol_partial(
                node.form_dict_str, node.problem_str, node.this_step
            )
            response = chat_gpt(user_prompt=prompt_to_gpt, n_used=n_eval, engine_used=engine_used)
            all_score = 0
            for i in range(n_eval):
                content = response.choices[i].message.content
                all_score += extract_score_final(content)
            final_score = all_score / (100 * n_eval)
            success = True
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            continue
    return final_score


def obtain_score_from_node(node, engine_used, n_eval):
    success = False
    while not success:
        try:
            prompt_to_gpt = obtain_prompt_solution_score_sol(node.form_dict_str, node.problem_str)
            response = chat_gpt(user_prompt=prompt_to_gpt, n_used=n_eval, engine_used=engine_used)
            all_score = 0
            for i in range(n_eval):
                content = response.choices[i].message.content
                all_score += extract_score_final(content)
            final_score = all_score / (100 * n_eval)
            success = True
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            continue
    return final_score


def filter_path(path):
    """
    Keep only path components relevant to the variable/constraint steps.
    Behavior preserved.
    """
    keywords = ["parameters", "decision_variables", "objective", "equality_constraints", "inequality_constraints"]
    path_components = path.split("/")
    filtered_components = [comp for comp in path_components if any(kw in comp for kw in keywords)]
    filtered_path = "/".join(filtered_components)
    return filtered_path


# =========================
# Rewards
# =========================
class Reward:
    """Mutable container for a scalar reward with a visit count (unchanged)."""

    def __init__(self, reward: float = 0):
        self.reward = reward
        self.count = 0

    def update_reward(self, new_reward: float):
        self.reward = new_reward

    def get_reward(self) -> float:
        return self.reward

    def increase_count(self):
        self.count += 1


class RewardManager:
    """
    Ranking/score-based reward manager (structure preserved).
    """

    def __init__(self, engine_used="GPT4o", comparison_type="rank", n_reward=N_REWARD):
        self.rankings = {}
        self.engine_used = engine_used
        self.comparison_type = comparison_type
        self.n_reward = n_reward

    def get_score(self, obj):
        prompt_to_gpt = obtain_prompt_solution_score(obj)
        response = chat_gpt(user_prompt=prompt_to_gpt, n_used=self.n_reward, engine_used=self.engine_used)
        all_score = 0
        for i in range(self.n_reward):
            content = response.choices[i].message.content
            all_score += extract_score_final(content)
        return all_score / (100 * self.n_reward)

    def compare_two_objects(self, obj1, obj2):
        """
        Compares two objects and returns the better one.
        Behavior preserved (including the known .values attribute access below).
        """
        total_solution = 2
        prompt_to_gpt = obtain_prompt_solution_compar(obj1, obj2)
        step = "possible_solution"
        response = chat_gpt(user_prompt=prompt_to_gpt, n_used=self.n_reward, engine_used=self.engine_used)
        current_dict = {f"{step}_{idx}": 0 for idx in range(1, total_solution + 1)}
        current_dict_other = {f"{step}_{idx}": 0 for idx in range(2, total_solution + 1)}

        for i in range(self.n_reward):
            content = response.choices[i].message.content
            dict_rank = extract_dictionary_rank_final(content, total_solution, step)
            try:
                if dict_rank is not None:
                    current_dict = update_rank(current_dict, dict_rank)
            except Exception:
                continue

        solution_this_object = current_dict["possible_solution_1"]
        # NOTE: Original behavior uses `.values` (no call), which we preserve.
        for this_value in current_dict_other.values:
            if this_value > solution_this_object:
                return False
        return True

    def get_reward_update_ranking(self, new_object):
        if new_object in self.rankings.keys():
            new_object.reward_backpropagation.increase_count()
            return None
        else:
            if self.comparison_type == "rank":
                self.update_values(new_object)
                self.update_rank_objects()
                new_object.reward_backpropagation.increase_count()
                return new_object.reward_backpropagation
            elif self.comparison_type == "score":
                this_score = self.get_score(new_object)
                new_object.reward_backpropagation.update_reward(this_score)

    def update_values(self, new_object):
        """
        Inserts new_object using a binary-search-like policy to minimize comparisons.
        Behavior preserved.
        """
        if not self.rankings:
            self.rankings[new_object] = 1
            return self.rankings

        ranked_objects = list(self.rankings.keys())
        low, high = 0, len(ranked_objects) - 1
        insert_index = len(ranked_objects)

        while low <= high:
            mid = (low + high) // 2
            comparison_result = self.compare_two_objects(new_object, ranked_objects[mid])

            if comparison_result == new_object:
                insert_index = mid
                high = mid - 1
            else:
                low = mid + 1

        ranked_objects.insert(insert_index, new_object)
        self.rankings = {obj: rank + 1 for rank, obj in enumerate(ranked_objects)}
        return self.rankings

    def update_rank_objects(self):
        lin = np.linspace(0, 1, len(self.rankings) + 2)[::-1]
        for obj, rank in self.rankings.items():
            this_score = lin[rank]
            obj.reward_backpropagation.update_reward(this_score)
