import os
import re
import pandas as pd
import numpy as np
import json

def collect_statistics(problem_id, root_folder):
    import os
    import re
    import pandas as pd

    # Define the types in order
    types_order = ['parameters', 'decision_variables', 'objective', 'equality_constraints', 'inequality_constraints']

    # Initialize a list to collect data
    data = []

    # Start from the problem_id folder
    root_path = os.path.join(root_folder, problem_id)

    # Get the only correct child of problem_id, which is '%d-parameters'
    # So we look for folders matching the pattern '%d-parameters'

    def traverse_node(current_path, current_type_index):
        # current_path: path to the current folder
        # current_type_index: index in types_order of the current type

        # Get the current type
        current_type = types_order[current_type_index]

        # Extract the type_id from the folder name
        folder_name = os.path.basename(current_path)
        match = re.match(r'(\d+)-' + current_type + '$', folder_name)
        if match:
            type_id = int(match.group(1))
        else:
            # Could not extract type_id, so we might have an incorrect folder
            return

        # Initialize counters
        num_child_correct = 0
        num_child_clustered = 0
        num_child_incorrect = 0

        # If we are at the last type, we do not have any further children
        if current_type_index == len(types_order) - 1:
            # Append the data and return
            data.append({
                'problem_id': problem_id,
                'type': current_type,
                'type_id': type_id,
                'num_child_correct': num_child_correct,
                'num_child_clustered': num_child_clustered,
                'num_child_incorrect': num_child_incorrect
            })
            return

        # The next type in the hierarchy
        next_type = types_order[current_type_index + 1]

        # List the children folders
        if not os.path.isdir(current_path):
            return

        children = os.listdir(current_path)

        # Process each child
        for child in children:
            child_path = os.path.join(current_path, child)
            if os.path.isdir(child_path):
                # Check if child is correct
                if re.match(r'\d+-' + next_type + '$', child):
                    num_child_correct += 1
                    # Recursively process the child
                    traverse_node(child_path, current_type_index + 1)
                elif 'z-clustered' in child:
                    num_child_clustered += 1
                elif 'z-incorrect' in child or 'z-empty' in child:
                    num_child_incorrect += 1
                else:
                    # Other folder names are ignored
                    pass

        # Append the data for the current folder
        data.append({
            'problem_id': problem_id,
            'type': current_type,
            'type_id': type_id,
            'num_child_correct': num_child_correct,
            'num_child_clustered': num_child_clustered,
            'num_child_incorrect': num_child_incorrect
        })

    # Start traversal
    # Find the '%d-parameters' folder under root_path
    if not os.path.isdir(root_path):
        print(f"Root path {root_path} is not a directory")
        return pd.DataFrame()

    # List the children of the root_path
    root_children = os.listdir(root_path)
    parameters_folder = None

    for child in root_children:
        child_path = os.path.join(root_path, child)
        if os.path.isdir(child_path):
            if re.match(r'\d+-parameters$', child):
                parameters_folder = child_path
                break

    if parameters_folder is None:
        print("No parameters folder found under problem_id")
        return pd.DataFrame()

    # Start traversal from parameters_folder
    traverse_node(parameters_folder, 0)  # 0 corresponds to 'parameters' in types_order

    # Create DataFrame
    df = pd.DataFrame(data)

    # Return DataFrame
    return df


import os
import re
from collections import OrderedDict

def collect_results_ordered_with_pairs(type_difficulty_pairs, folder1, folder2):
    results = {}  # Dictionary to store problem_id: result
    problems_with_results = []
    problems_not_found = []
    found_in_folder = []
    result_pairs = []  # List to store (type, difficulty) pairs

    # Helper function to extract numeric part from folder name
    def extract_problem_id(folder_name):
        match = re.search(r'\d+', folder_name)
        return int(match.group()) if match else float('inf')  # Use float('inf') to push unknowns to the end

    # Collect all data first
    for q_type, difficulty in type_difficulty_pairs:
        # Construct the folder paths for both folders
        folder1_path = os.path.join(folder1, f"{q_type}_{difficulty}")
        folder2_path = os.path.join(folder2, f"{q_type}_{difficulty}")

        # Get problem ids based on available folders in both folder1 and folder2
        folder1_ids = os.listdir(folder1_path) if os.path.exists(folder1_path) else []
        folder2_ids = os.listdir(folder2_path) if os.path.exists(folder2_path) else []
        # Combine and sort problem_ids within this pair; overall sorting will be handled later
        problem_ids = sorted(set(folder1_ids) | set(folder2_ids), key=extract_problem_id)

        for problem_id in problem_ids:
            found = False
            problem_folder1 = os.path.join(folder1_path, problem_id)
            problem_folder2 = os.path.join(folder2_path, problem_id)

            # Try folder1 first
            result_path1 = os.path.join(problem_folder1, "all_results.jsonl")
            if os.path.exists(result_path1):
                try:
                    with open(result_path1, 'r') as f:
                        results[problem_id] = f.read()
                    problems_with_results.append(problem_id)
                    found_in_folder.append(folder1)
                    result_pairs.append((q_type, difficulty))
                    found = True
                except Exception as e:
                    print(f"Error reading {result_path1}: {e}")

            # If not found in folder1, try folder2
            if not found:
                result_path2 = os.path.join(problem_folder2, "all_results.jsonl")
                if os.path.exists(result_path2):
                    try:
                        with open(result_path2, 'r') as f:
                            results[problem_id] = f.read()
                        problems_with_results.append(problem_id)
                        found_in_folder.append(folder2)
                        result_pairs.append((q_type, difficulty))
                        found = True
                    except Exception as e:
                        print(f"Error reading {result_path2}: {e}")
                else:
                    problems_not_found.append(problem_id)

    # Now, sort 'problems_with_results' numerically and reorder other lists accordingly
    def get_numeric_id(problem_id):
        """Extracts the numeric part from a problem ID string like 'problem_40'."""
        match = re.search(r'\d+', problem_id)
        return int(match.group()) if match else float('inf')

    # Create a list of tuples containing all related data
    zipped_data = list(zip(problems_with_results, found_in_folder, result_pairs))

    # Sort the zipped data based on the numeric part of problem_id
    sorted_zipped_data = sorted(zipped_data, key=lambda x: get_numeric_id(x[0]))

    # Unzip the sorted data back into separate lists
    sorted_problems_with_results, sorted_found_in_folder, sorted_result_pairs = zip(*sorted_zipped_data) if sorted_zipped_data else ([], [], [])

    # Convert tuples back to lists
    sorted_problems_with_results = list(sorted_problems_with_results)
    sorted_found_in_folder = list(sorted_found_in_folder)
    sorted_result_pairs = list(sorted_result_pairs)

    # Create an OrderedDict for 'results' based on the sorted problem IDs
    sorted_results = OrderedDict()
    for problem_id in sorted_problems_with_results:
        sorted_results[problem_id] = results[problem_id]

    # Optionally, sort 'problems_not_found' numerically as well
    sorted_problems_not_found = sorted(problems_not_found, key=get_numeric_id)

    return {
        "results": sorted_results,  # OrderedDict maintains the sorted order
        "problems_with_results": sorted_problems_with_results,  # Numerically sorted list
        "problems_not_found": sorted_problems_not_found,  # Numerically sorted list
        "found_in_folder": sorted_found_in_folder,  # Aligned with sorted_problems_with_results
        "result_pairs": sorted_result_pairs  # Aligned with sorted_problems_with_results
    }






import os
import re
import json
import pandas as pd

def get_results_dataframe(root_folder, problem_id):
    problem_folder = os.path.join(root_folder, problem_id)
    if not os.path.isdir(problem_folder):
        print(f"Problem folder {problem_folder} does not exist.")
        return pd.DataFrame(columns=["problem", "path", "predicted", "python_string"])
    
    # Load 'all_results_path.json'
    results_json_path = os.path.join(problem_folder, 'all_results_path.json')
    if not os.path.isfile(results_json_path):
        print(f"'all_results_path.json' not found in {problem_folder}.")
        return pd.DataFrame(columns=["problem", "path", "predicted", "python_string"])
    
    with open(results_json_path, 'r') as f:
        results_dict = json.load(f)
    
    # Find all leaf nodes (folders ending with 'inequality_constraints')
    leaf_nodes = []
    
    def find_leaf_nodes(current_path, path_accum):
        if not os.path.isdir(current_path):
            return
        for entry in os.listdir(current_path):
            full_entry = os.path.join(current_path, entry)
            if os.path.isdir(full_entry):
                # Parse entry to get idx and step
                m = re.match(r'(\d+)-(\w+)', entry)
                if m:
                    idx, step = m.group(1), m.group(2)
                    new_path_accum = path_accum + [(step, idx)]
                    if step == 'inequality_constraints':
                        # Leaf node
                        leaf_nodes.append((full_entry, new_path_accum))
                    else:
                        find_leaf_nodes(full_entry, new_path_accum)
    find_leaf_nodes(problem_folder, [])
    
    data = []
    
    def get_best_objective_from_json(json_dict, path_steps):
        current_dict = json_dict
        for step, idx in path_steps:
            if step == 'parameters':
                continue  # Skip 'parameters' step as it's not in JSON keys
            # Build the regex pattern to match the key, ignoring the '(R: X)' part
            pattern = rf'^{step} \(idx: {idx}\)(?: \(R: [\d\.]+\))?$'
            found = False
            for key in current_dict.keys():
                if re.match(pattern, key):
                    current_dict = current_dict[key]
                    found = True
                    break
            if not found:
                # Cannot find matching key
                return None
        # After traversing, get 'best_objective' from current_dict
        return current_dict.get('best_objective', None)
    
    for leaf_node_path, path_accum in leaf_nodes:
        # Read 'A-python_runnable.py'
        python_file_path = os.path.join(leaf_node_path, 'A-python_runnable.py')
        if os.path.isfile(python_file_path):
            with open(python_file_path, 'r') as f:
                python_string = f.read()
        else:
            python_string = ''
        
        json_file_path = os.path.join(leaf_node_path, 'A-form-eval-string.json')


        if os.path.isfile(json_file_path):
            with open(json_file_path, 'r') as file:
                data_json = json.load(file)
        else:
            data_json = {}
        

        # Adjust path_accum to skip 'parameters' steps
        path_steps = [(step, idx) for step, idx in path_accum if step != 'parameters']
        predicted = get_best_objective_from_json(results_dict, path_steps)
        
        # Build the relative path from 'problem_id' folder
        relative_path = os.path.relpath(leaf_node_path, problem_folder)
        
        data.append({
            'problem': problem_id,
            'path': relative_path,
            'predicted': predicted,
            'python_string': python_string,
            'data_json': data_json
        })

    df = pd.DataFrame(data, columns=["problem", "path", "predicted", "python_string", "data_json"])
    return df


def collect_results(folder_priority, f, list_of_tuples=None, root_folder='root_folder', is_df = True):
    """
    Collect results from specified folders based on priority and problem IDs.

    Parameters:
    - folder_priority: List of folder names in priority order.
    - f: Function that takes two inputs: the base folder and the problem_id (folder).
    - list_of_tuples: List of tuples (first, second) or None.
    - root_folder: The root directory containing the folders.

    Returns:
    - A pandas DataFrame containing the collected results.
    """
    if is_df: 
        final_df = pd.DataFrame()
    else:
        final_df = []
    found_problem_ids = set()
    
    for possible_folder in folder_priority:
        folder_path = os.path.join(root_folder, possible_folder)
        
        if list_of_tuples is None:
            # Problem IDs are directly inside the possible_folder
            if os.path.isdir(folder_path):
                problem_ids = [d for d in os.listdir(folder_path)
                               if os.path.isdir(os.path.join(folder_path, d))]
            else:
                problem_ids = []
        else:
            # Build subfolders from list_of_tuples
            subfolders = [f"{first}_{second}" for first, second in list_of_tuples]
            problem_ids = []
            for subfolder in subfolders:
                subfolder_path = os.path.join(folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    problem_ids.extend([d for d in os.listdir(subfolder_path)
                                        if os.path.isdir(os.path.join(subfolder_path, d))])
        
        for problem_id in problem_ids:
            if problem_id in found_problem_ids:
                continue
            
            if list_of_tuples is None:
                # Path for problem_id directly inside possible_folder
                base_folder = folder_path
            else:
                # Path for problem_id inside a subfolder (e.g., 'possible_folder/Medium_IP')
                for subfolder in subfolders:
                    base_folder = os.path.join(folder_path, subfolder)
                    if os.path.exists(os.path.join(base_folder, problem_id)):
                        break
            
            if not os.path.exists(os.path.join(base_folder, problem_id)):
                continue
            
            # Call f with the base_folder and problem_id
            df = f(base_folder, problem_id)
            if is_df:
                if not df.empty:
                    final_df = pd.concat([final_df, df], ignore_index=True)
                    found_problem_ids.add(problem_id)
            else:
                if len(df) != 0:
                    final_df += [df]

    return final_df


def get_all_results_json(base_folder, problem_id):
    results_folder = os.path.join(base_folder, problem_id)
    results_file   = f'{results_folder}/all_results_path.json'
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except:
        return {}

def evaluate_predictions(df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """
    This function takes in a pandas DataFrame `df` and a `tolerance` value.
    It computes if a prediction is correct based on the condition:
    - abs(predicted - ground_truth)/ground_truth < tolerance (if ground_truth != 0)
    - abs(predicted) < tolerance (if ground_truth == 0)
    
    Args:
    - df (pd.DataFrame): DataFrame with columns 'predicted' and 'en_answer'.
    - tolerance (float): Tolerance threshold to determine correctness.
    
    Returns:
    - pd.DataFrame: Updated DataFrame with a new column 'is_correct'.
    """
    def check_correctness(row, tolerance):
        try:
            predicted = float(row['predicted'])
            ground_truth = float(row['en_answer'])
            if ground_truth == 0:
                return abs(predicted) < tolerance
            else:
                return abs(predicted - ground_truth) / ground_truth < tolerance
        except:
            return False  # If conversion to float fails, mark as incorrect

    df['is_correct'] = df.apply(lambda row: check_correctness(row, tolerance), axis=1)
    return df


def get_problems_with_correct_predictions(df: pd.DataFrame) -> list:
    """
    This function returns a unique list of 'problem' values from the DataFrame
    that have at least one correct prediction (where 'is_correct' is True).
    
    Args:
    - df (pd.DataFrame): DataFrame with columns 'problem' and 'is_correct'.
    
    Returns:
    - list: A unique list of 'problem' values that have at least one correct prediction.
    """
    correct_problems = df[df['is_correct'] == True]['problem'].unique().tolist()
    return correct_problems, df['problem'].unique().tolist()


def create_subset(df, n):
    # Compute counts of correct responses per problem_id
    correct_counts = df[df['is_correct'] == True].groupby('problem').size()

    # Compute counts of unique incorrect 'predicted' values per problem_id
    incorrect_counts = df[df['is_correct'] == False].groupby('problem')['predicted'].nunique()

    # Identify problem_ids that meet the criteria
    valid_problem_ids = correct_counts[correct_counts >= 1].index.intersection(
        incorrect_counts[incorrect_counts >= n].index
    )

    # List to collect DataFrames
    dfs = []

    # Process each valid problem_id
    for pid in valid_problem_ids:
        pid_df = df[df['problem'] == pid]

        # Select one correct response
        correct_responses = pid_df[pid_df['is_correct'] == True]
        correct_row = correct_responses.sample(n=1, random_state=42).copy()
        correct_row['specific_id'] = 0

        # Get unique incorrect responses
        incorrect_responses = pid_df[pid_df['is_correct'] == False]
        unique_incorrect_responses = incorrect_responses.drop_duplicates(subset=['predicted'])

        # Sample n unique incorrect responses
        incorrect_rows = unique_incorrect_responses.sample(n=n, random_state=42).copy()
        incorrect_rows['specific_id'] = range(1, n + 1)

        # Combine correct and incorrect responses
        pid_subset = pd.concat([correct_row, incorrect_rows])

        # Append to list
        dfs.append(pid_subset)

    # Concatenate all subsets
    result_df = pd.concat(dfs, ignore_index=True)

    #return result_df


    return result_df, result_df['problem'].unique().tolist()


import pandas as pd
import itertools

# Updated comparison function
# def compare_solutions(s1, s2, num_comparisons):
#     # Simulate comparison results
#     # Replace this with your actual comparison logic
#     import random
#     wins_s1 = sum(random.random() < 0.5 for _ in range(num_comparisons))
#     wins_s2 = num_comparisons - wins_s1
#     return wins_s1, wins_s2

def extract_dictionary_rank_final(text):
    # Use regex to find the dictionary part, making sure we handle optional quotes around values
    dict_pattern = re.search(r'rank\s*=\s*\{(.*?)\}', text, re.DOTALL)
    if dict_pattern:
        dict_str = dict_pattern.group(1).strip()
        # Handle cases where values are enclosed in quotes or not
        key_value_pairs = re.findall(r'(\d+):\s*"?(possible_solution_\d+)"?', dict_str)
        # Create the dictionary with keys and values converted to strings
        extracted_dict = {str(key): str(value) for key, value in key_value_pairs}
        return extracted_dict
    else:
        return {}

from utils import chat_gpt
N_USED_FINAL = 5

def obtain_prompt_solution_compar_string(sol1, sol2, problem_str):
    from prompts import INST_RANK_FINAL
    this_prompt = INST_RANK_FINAL.replace("###PROBLEM DESCRIPTION###", problem_str)
    this_prompt = this_prompt.replace("possible_solution_1 = {}", f'possible_solution_1 = \\\n"""{sol1}\n"""')
    this_prompt = this_prompt.replace("possible_solution_2 = {}", f'possible_solution_2 = \\\n"""{sol2}\n"""')
    return this_prompt

def compare_solutions(s1, s2, num_comparisons, problem):
    new_prompt = obtain_prompt_solution_compar_string(s1, s2, problem)
    response   = chat_gpt(user_prompt = new_prompt, n_used = num_comparisons, engine_used = 'GPT4o')
    wins_s1, wins_s2 = 0, 0
    for i in range(num_comparisons):
        text          = response.choices[i].message.content
        this_dict     = extract_dictionary_rank_final(text)
        new_this_dict = {int(key): value for key, value in this_dict.items()}
        try:
            if new_this_dict[1] == 'possible_solution_1':
                wins_s1 += 1
            else:
                wins_s2 += 1
        except:
            import pdb
            pdb.set_trace()
            print("A")
    return wins_s1, wins_s2

def extract_dictionary_rank_final_form(text):
    # Use regex to find the dictionary part, making sure we handle different formats and quote types around values
    dict_pattern = re.search(r'rank\s*=\s*\{(.*?)\}', text, re.DOTALL)
    if dict_pattern:
        dict_str = dict_pattern.group(1).strip()
        # Adjust regex to handle single quotes, double quotes, or no quotes around the solutions
        key_value_pairs = re.findall(r'(\d+):\s*["\'\s]?(A|B)["\'\s]?', dict_str)
        # Create the dictionary with keys and values converted to strings
        extracted_dict = {str(key): str(value) for key, value in key_value_pairs}
        return extracted_dict
    else:
        return {}
    
def extract_dictionary_rank_final_form_2(text):
    # Use regex to find the dictionary part, making sure we handle different formats and quote types around values
    dict_pattern = re.search(r'best_solution\s*=\s*\{(.*?)\}', text, re.DOTALL)
    if dict_pattern:
        dict_str = dict_pattern.group(1).strip()
        # Adjust regex to handle single quotes, double quotes, or no quotes around the solutions
        key_value_pairs = re.findall(r'(\d+):\s*["\'\s]?(solution_A|solution_B)["\'\s]?', dict_str)
        # Create the dictionary with keys and values converted to strings
        extracted_dict = {str(key): str(value) for key, value in key_value_pairs}
        return extracted_dict
    else:
        return {}
    
def obtain_prompt_solution_compar_string_form(sol1, sol2, problem_str):
    from prompts import INST_RANK_FINAL_STEP
    this_prompt = INST_RANK_FINAL_STEP.replace("###PROBLEM DESCRIPTION###", problem_str)
    this_prompt = this_prompt.replace("possible_solution_A = {}", f'possible_solution_A = \\\n"""{str(sol1)}\n"""')
    this_prompt = this_prompt.replace("possible_solution_B = {}", f'possible_solution_B = \\\n"""{str(sol2)}\n"""')
    return this_prompt


def obtain_prompt_solution_compar_string_form_2(sol1, sol2, problem_str):
    from prompts import INST_RANK_FINAL_STEP_2
    this_prompt = INST_RANK_FINAL_STEP_2.replace("###PROBLEM DESCRIPTION###", problem_str)
    this_prompt = this_prompt.replace("solution_A = {}", f'solution_A = \\\n"""{str(sol1)}\n"""')
    this_prompt = this_prompt.replace("solution_B = {}", f'solution_B = \\\n"""{str(sol2)}\n"""')
    return this_prompt

import re
def extract_dictionary_rank_final_form_2(text):
    # Use regex to find the dictionary part, making sure we handle different formats and quote types around values
    dict_pattern = re.search(r'best_solution\s*=\s*\{(.*?)\}', text, re.DOTALL)
    if dict_pattern:
        dict_str = dict_pattern.group(1).strip()
        # Adjust regex to handle single quotes, double quotes, or no quotes around the solutions
        key_value_pairs = re.findall(r'(\d+):\s*["\'\s]?(solution_A|solution_B)["\'\s]?', dict_str)
        # Create the dictionary with keys and values converted to strings
        extracted_dict = {str(key): str(value) for key, value in key_value_pairs}
        return extracted_dict
    else:
        return {}
    
def compare_solutions_form(s1, s2, num_comparisons, problem, engined_used = 'GPT4o'):
    new_prompt = obtain_prompt_solution_compar_string_form_2(s2, s1, problem)
    response   = chat_gpt(user_prompt = new_prompt, n_used = num_comparisons, engine_used = engined_used)
    wins_s1, wins_s2 = 0, 0
    for i in range(num_comparisons):
        text          = response.choices[i].message.content
        this_dict     = extract_dictionary_rank_final_form_2(text)
        new_this_dict = {int(key): value for key, value in this_dict.items()}
        #try:
        if new_this_dict[1] == 'solution_B':
            wins_s1 += 1
        else:
            wins_s2 += 1
    total_score = wins_s1 + wins_s2
    return wins_s1 / total_score, wins_s2 / total_score

def obtain_prompt_solution_score(sol1, problem_str):
    from prompts import INST_SCORE_FINAL
    this_prompt = INST_SCORE_FINAL.replace("###PROBLEM DESCRIPTION###", problem_str)
    this_prompt = this_prompt.replace("possible_solution = {}", f'possible_solution = \\\n"""{str(sol1)}\n"""')
    #this_prompt = this_prompt.replace("possible_solution_B = {}", f'possible_solution_B = \\\n"""{str(sol2)}\n"""')
    return this_prompt

def obtain_prompt_solution_score_sol(sol1, problem_str):
    from prompts import INST_SCORE_FINAL
    this_prompt = INST_SCORE_FINAL.replace("###PROBLEM DESCRIPTION###", problem_str)
    this_prompt = this_prompt.replace("possible_solution = {}", f'possible_solution = \\\n"""{str(sol1)}\n"""')
    #this_prompt = this_prompt.replace("possible_solution_B = {}", f'possible_solution_B = \\\n"""{str(sol2)}\n"""')
    return this_prompt


# def obtain_prompt_solution_score_sol_partial(sol1, problem_str):
#     from prompts import INST_SCORE_FINAL_PARTIAL
#     this_prompt = INST_SCORE_FINAL_PARTIAL.replace("###PROBLEM DESCRIPTION###", problem_str)
#     this_prompt = this_prompt.replace("possible_solution = {}", f'possible_solution = \\\n"""{str(sol1)}\n"""')
#     #this_prompt = this_prompt.replace("possible_solution_B = {}", f'possible_solution_B = \\\n"""{str(sol2)}\n"""')
#     return this_prompt


def obtain_prompt_solution_score_sol_partial(formalization_dict_str, problem_description, this_step):
    from prompts import INST_SCORE_FINAL_PARTIAL
    prompt = INST_SCORE_FINAL_PARTIAL.replace("###PROBLEM DESCRIPTION###", problem_description).replace("#VARIABLE#", this_step)
    for par in formalization_dict_str.keys():
        prompt = prompt.replace("'%s': {}," % par, "'%s': %s," % (par, formalization_dict_str[par]))
    #prompt = prompt.replace("possible_solution = {}", f'possible_solution = \\\n"""{str(sol1)}\n"""')
    return prompt 


import re
def extract_score_prompt(input_string):
    # Regular expression to find the float or integer value after "score = {"
    score_match = re.search(r'score\s*=\s*{\s*"possible_solution"\s*:\s*([\d.]+)\s*}', input_string)
    
    if score_match:
        # Convert the matched value to a float
        score = float(score_match.group(1))
        return score
    else:
        raise ValueError("Score not found in the string.")

def compare_solutions_form_score(s1, s2, num_comparisons, problem, engined_used = 'GPT4o'):
    new_prompt_1 = obtain_prompt_solution_score(s1, problem)
    response_1   = chat_gpt(user_prompt = new_prompt_1, n_used = num_comparisons, engine_used = engined_used)

    #import pdb
    #pdb.set_trace()
    new_prompt_2 = obtain_prompt_solution_score(s2, problem)
    response_2   = chat_gpt(user_prompt = new_prompt_2, n_used = num_comparisons, engine_used = engined_used)

    wins_s1, wins_s2 = 0, 0
    for i in range(num_comparisons):
        text_1        = response_1.choices[i].message.content
        text_2        = response_2.choices[i].message.content

        wins_s1 += extract_score_prompt(text_1) * 1.0
        wins_s2 += extract_score_prompt(text_2) * 1.0


    total_score = wins_s1 + wins_s2
    return wins_s1 / total_score, wins_s2 / total_score

# Modified compute_general_scores function
def compute_general_scores(df, num_comparisons, output_folder, modality = 'python-rank'):
    df_list = []
    #comparisons = {}  # To store already computed comparisons
    #MAX_RETRIES = 3
    DEFAULT_WINS_I = 0
    DEFAULT_WINS_J = 0
    MAX_RETRIES = 3
    import logging

    # Configure logging to capture errors
    logging.basicConfig(
        filename='comparison_errors.log',
        level=logging.ERROR,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for problem_id, group_df in df.groupby('problem'):

        output_folder_problem = os.path.join(output_folder, f'{problem_id}')
        os.makedirs(output_folder_problem, exist_ok=True)
        output_problem_results = os.path.join(output_folder_problem, f'all_results.csv')
        print("problem_id, ", problem_id)
        print("folder exists ", output_problem_results, " ", os.path.exists(output_problem_results))

        if os.path.exists(output_problem_results):
            df_list.append(pd.read_csv(output_problem_results))
            continue

        comparisons = {}  # To store already computed comparisons
        if  modality == 'python-rank':
            id_to_code = dict(zip(group_df['specific_id'], group_df['python_string']))
        elif  modality == 'form-rank' or modality == 'form-score':
            id_to_code = dict(zip(group_df['specific_id'], group_df['data_json']))

        problem_description = group_df['en_question'].iloc[0] 
        ids = list(id_to_code.keys())
        total_wins = {i: 0 for i in ids}
        comparison_results = {i: {} for i in ids}

        for i in ids:
            for j in ids:
                if i == j:
                    continue  # Skip comparison with self

                # Create a consistent key to avoid redundant comparisons
                key = (min(i, j), max(i, j))

                if key not in comparisons:
                    s_i = id_to_code[i]
                    s_j = id_to_code[j]
                    print("problem AAAAA: ", problem_id)

                    success = False
                    retries = 0

                    while not success and retries < MAX_RETRIES:
                        try:
                            if modality == 'python-rank':
                                wins_i, wins_j = compare_solutions(s_i, s_j, num_comparisons, problem_description)
                            elif modality == 'form-rank':
                                wins_i, wins_j = compare_solutions_form(s_i, s_j, num_comparisons, problem_description)
                            elif modality == 'form-score':
                                wins_i, wins_j = compare_solutions_form_score(s_i, s_j, num_comparisons, problem_description)
                            success = True  # If no exception, mark as successful
                        except Exception as e:
                            retries += 1
                            logging.error(f"Comparison failed for IDs ({i}, {j}) on attempt {retries}: {e}")
                            if retries >= MAX_RETRIES:
                                logging.error(f"Max retries reached for IDs ({i}, {j}). Assigning default win values.")
                                wins_i, wins_j = DEFAULT_WINS_I, DEFAULT_WINS_J
                            else:
                                logging.info(f"Retrying comparison for IDs ({i}, {j})...")

                    # if not is_form_string:
                    #     wins_i, wins_j = compare_solutions(s_i, s_j, num_comparisons, problem_description)
                    # else:
                    #     wins_i, wins_j = compare_solutions_form(s_i, s_j, num_comparisons, problem_description)
                    comparisons[key] = {i: wins_i, j: wins_j}
                else:
                    # Retrieve the stored comparison results
                    result = comparisons[key]
                    wins_i = result[i]
                    wins_j = result[j]

                # Update total wins
                total_wins[i] += wins_i
                total_wins[j] += wins_j

                # Store individual comparison results
                comparison_results[i][f'compare_with_{j}'] = wins_i
                comparison_results[j][f'compare_with_{i}'] = wins_j

        # Create DataFrame for comparison results
        comparison_df = pd.DataFrame.from_dict(comparison_results, orient='index')

        # Create Series for general scores
        general_score_series = pd.Series(total_wins, name='general_score')

        # Merge with the original group DataFrame
        group_df = group_df.set_index('specific_id')
        group_df = group_df.join(comparison_df)
        group_df = group_df.join(general_score_series)
        group_df = group_df.reset_index()

        df_list.append(group_df)

        compare_columns = ['specific_id'] + [col for col in group_df.columns if col.startswith('compare_with_')]  + ['general_score']
        comparison_data = group_df[compare_columns]
        # Save to CSV file
        output_file_comparisons = os.path.join(output_folder, f'comparisons_{problem_id}.jsonl')

        comparison_data.to_json(output_file_comparisons, orient='records', lines=True)

        group_df.to_csv(output_problem_results, index=False)

        for index, row in group_df.iterrows():
            if modality == 'python-rank':
                code_str             = row['python_string']
            elif modality == 'form-rank' or modality == 'form-score':
                code_str             = str(row['data_json'])
            this_specific_id     = row['specific_id']
            output_file_python   = os.path.join(output_folder_problem, f'python_{this_specific_id}.py')
            with open(output_file_python, "w") as script_file:
                script_file.write(code_str)

    # Concatenate all groups
    result_df = pd.concat(df_list, ignore_index=True)
    return result_df


def compute_ranking(pairwise_comparisons):
    # Prepare the data
    data = []
    labels = []
    
    # Extract the valid solution IDs
    solutions = sorted(set(pairwise_comparisons.keys()) | 
                       set(j for sub in pairwise_comparisons.values() for j in sub))
    
    # Iterate over all unique pairs without duplication
    for i in solutions:
        for j in solutions:
            if i < j:  # Avoid duplicate pairs and self-comparison
                # Get the score of i against j
                score_ij = pairwise_comparisons.get(i, {}).get(j, None)
                score_ji = pairwise_comparisons.get(j, {}).get(i, None)
    
                if score_ij is not None:
                    if not (0 <= score_ij <= 1):
                        raise ValueError(f"Invalid score {score_ij} for comparison ({i}, {j}). Scores must be between 0 and 1.")
                    if score_ij > 0.5:
                        # i beats j
                        data.append([i, j])
                        labels.append(1)
                    elif score_ij < 0.5:
                        # i loses to j
                        data.append([i, j])
                        labels.append(0)
                    # If score is exactly 0.5, we skip it (tie)
                elif score_ji is not None:
                    if not (0 <= score_ji <= 1):
                        raise ValueError(f"Invalid score {score_ji} for comparison ({j}, {i}). Scores must be between 0 and 1.")
                    if score_ji > 0.5:
                        # j beats i, so i loses to j
                        data.append([i, j])
                        labels.append(0)
                    elif score_ji < 0.5:
                        # j loses to i, so i beats j
                        data.append([i, j])
                        labels.append(1)
                    # If score is exactly 0.5, we skip it (tie)
                # If both scores are None or exactly 0.5, we skip the pair
    
    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    # Check that labels contain both 0 and 1
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Insufficient variation in labels. The model requires both wins and losses.")
    
    # Create feature vectors
    n_solutions = max(solutions) + 1  # Determine the maximum solution index
    X = np.zeros((len(data), n_solutions))
    for idx, (i, j) in enumerate(data):
        X[idx, i] = 1
        X[idx, j] = -1
    
    # Fit the Bradley-Terry model
    model = LogisticRegression(fit_intercept=False, solver='lbfgs')
    model.fit(X, labels)
    
    # Extract abilities
    abilities = model.coef_[0]
    ranking = [solution for solution in np.argsort(-abilities) if solution in solutions]
    
    # Display the ranking
    print("Final Ranking:")
    for rank, solution in enumerate(ranking, 1):
        print(f"Rank {rank}: Solution {solution} with ability score {abilities[solution]:.2f}")
    
    return ranking, abilities


import numpy as np
from sklearn.linear_model import LogisticRegression


def compute_ranking(pairwise_comparisons):
    # Prepare the data
    data = []
    labels = []
    
    # Extract the valid solution IDs
    solutions = sorted(set(pairwise_comparisons.keys()) | 
                       set(j for sub in pairwise_comparisons.values() for j in sub))
    
    # Iterate over all unique pairs without duplication
    for i in solutions:
        for j in solutions:
            if i < j:  # Avoid duplicate pairs and self-comparison
                # Get the score of i against j
                score_ij = pairwise_comparisons.get(i, {}).get(j, None)
                score_ji = pairwise_comparisons.get(j, {}).get(i, None)
    
                if score_ij is not None:
                    if not (0 <= score_ij <= 1):
                        raise ValueError(f"Invalid score {score_ij} for comparison ({i}, {j}). Scores must be between 0 and 1.")
                    if score_ij > 0.5:
                        # i beats j
                        data.append([i, j])
                        labels.append(1)
                    elif score_ij < 0.5:
                        # i loses to j
                        data.append([i, j])
                        labels.append(0)
                    # If score is exactly 0.5, we skip it (tie)
                elif score_ji is not None:
                    if not (0 <= score_ji <= 1):
                        raise ValueError(f"Invalid score {score_ji} for comparison ({j}, {i}). Scores must be between 0 and 1.")
                    if score_ji > 0.5:
                        # j beats i, so i loses to j
                        data.append([i, j])
                        labels.append(0)
                    elif score_ji < 0.5:
                        # j loses to i, so i beats j
                        data.append([i, j])
                        labels.append(1)
                    # If score is exactly 0.5, we skip it (tie)
                # If both scores are None or exactly 0.5, we skip the pair
    
    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    # Check that labels contain both 0 and 1
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Insufficient variation in labels. The model requires both wins and losses.")
    
    # Create feature vectors
    n_solutions = max(solutions) + 1  # Determine the maximum solution index
    X = np.zeros((len(data), n_solutions))
    for idx, (i, j) in enumerate(data):
        X[idx, i] = 1
        X[idx, j] = -1
    
    # Fit the Bradley-Terry model
    model = LogisticRegression(fit_intercept=False, solver='lbfgs')
    model.fit(X, labels)
    
    # Extract abilities
    abilities = model.coef_[0]
    ranking = [solution for solution in np.argsort(-abilities) if solution in solutions]
    
    # Display the ranking
    print("Final Ranking:")
    for rank, solution in enumerate(ranking, 1):
        print(f"Rank {rank}: Solution {solution} with ability score {abilities[solution]:.2f}")
    
    return ranking, abilities