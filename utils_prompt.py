import re
def obtain_new_prompt(prompt, formalization_dict_str, instruction = ''):
    for par in formalization_dict_str.keys():
        prompt = prompt.replace("'%s': {}," % par, "'%s': %s," % (par, formalization_dict_str[par]))
    prompt += f"\n{instruction}"
    return prompt 

def obtain_new_prompt_dv(prompt, formalization_dict_str, instruction = '', n_top = 3):
    for par in formalization_dict_str.keys():
        prompt = prompt.replace("'%s': {}," % par, "'%s': %s," % (par, formalization_dict_str[par]))
    prompt += f"\n{instruction}"
    prompt = prompt.replace("##N_TOP##", n_top)
    return prompt 

def obtain_prompt_solution_compar(obj1, obj2):
    from prompts import INST_RANK_FINAL
    sol1, sol2 = obj1.load_python_script(), obj2.load_python_script() 
    this_prompt = INST_RANK_FINAL.replace("###PROBLEM DESCRIPTION###", obj1.problem_str)
    this_prompt = this_prompt.replace("possible_solution_1 = {}", f"possible_solution_1 = {sol1}")
    this_prompt = this_prompt.replace("possible_solution_2 = {}", f"possible_solution_2 = {sol2}")
    return this_prompt

def obtain_prompt_solution_compar_string(sol1, sol2, problem_str):
    from prompts import INST_RANK_FINAL
    this_prompt = INST_RANK_FINAL.replace("###PROBLEM DESCRIPTION###", problem_str)
    this_prompt = this_prompt.replace("possible_solution_1 = {}", f'possible_solution_1 = \n"""{sol1}\n"""')
    this_prompt = this_prompt.replace("possible_solution_2 = {}", f'possible_solution_2 = \n"""{sol2}\n"""')
    return this_prompt

def obtain_prompt_solution_score(obj):
    from prompts import INST_SCORE_FINAL
    sol = obj.load_python_script() 
    this_prompt = INST_SCORE_FINAL.replace("###PROBLEM DESCRIPTION###", obj.problem_str)
    this_prompt = this_prompt.replace("possible_solution = {}", f"possible_solution = \n'''{sol}'''")
    return this_prompt

def get_rank_filtering_prompt(prompt, problem_description, formalization_dict_str, solutions):
    prompt = prompt.replace("###PROBLEM DESCRIPTION###", problem_description)
    for par in formalization_dict_str.keys():
        prompt = prompt.replace("'%s': {}," % par, "'%s': %s," % (par, formalization_dict_str[par]))
    prompt = prompt.replace("solutions = {}", f"solutions = {solutions}" )
    return prompt 

import ast

def fix_single_equals(equality_constraints):
    """
    This function detects and replaces any instance of a single '=' (not part of '==')
    with '==' in all values of the provided dictionary.
    """
    fixed_constraints = {}
    
    for key, value in equality_constraints.items():
        # Replace '=' with '==' only if it's not already part of '=='
        fixed_value = value.replace(' = ', ' == ')
        fixed_constraints[key] = fixed_value
    
    return fixed_constraints


def check_none_key_value(d):
    return d.get(None) is None and None in d

def replace_all_dict_parameters(form_dict):
    final_form_dict = {}
    for key, value in form_dict.items():
        if key == 'parameters':
            final_form_dict[key] = value
            continue
        if (key == 'equality_constraints' or key == "inequality_constraints") and check_none_key_value(value):
            final_form_dict[key] = value
            continue
        final_form_dict[key] = replace_parameters_with_variable_dict(value)

    return final_form_dict

def detect_list_comprehension(s):
    s = s.strip()
    if not (s.startswith('[') and s.endswith(']')):
        #return s, False
        return s

    bracket_count = 0
    for i, char in enumerate(s):
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
        
        if bracket_count == 0 and i != len(s) - 1:
            #return s, False
            return s
    #return s, True
    return s[1:-1]

def eliminate_list_comprehension_brackets(this_dict):
    new_dict = {}
    for key, value in this_dict.items():
        new_dict[key] = detect_list_comprehension(value)
    return new_dict

def replace_parameters_with_variable(this_input, this_step):
    # Use regular expression to find patterns like parameters["variable"]
    # The pattern looks for 'parameters["' followed by any string (non-greedy), ending with '"]'
    if this_step != 'decision_variables':
        return re.sub(r'parameters\["(.*?)"\]', r'\1', this_input)
    else:
        if this_input['iteration_space'] is None:
            return this_input
        this_input['iteration_space'] = re.sub(r'parameters\["(.*?)"\]', r'\1', this_input['iteration_space'])
        return this_input

def not_contains_inequalities(s: str) -> bool:
    comparison_operations = ['<=', '< =', '>=', '> =', '<', '>']
    pattern = re.compile(r'<=|< =|>=|> =|<|>')
    if pattern.search(s):
        return False  # If found, return False
    return True  # If none found, return True

from utils_general import separate_constraint_from_for

def replace_inequalities(this_dict):
    new_dict = {}
    for key, value in this_dict.items():
        main_expr, for_loop = separate_constraint_from_for(value)
        if not_contains_inequalities(main_expr):
            new_dict[key] = value
    return new_dict

def replace_inequalities_not_equal(this_dict):
    new_dict = {}
    for key, value in this_dict.items():
        main_expr, for_loop = separate_constraint_from_for(value)
        if '>' in main_expr and not '>=' in main_expr and not '> =' in main_expr:
            new_dict[key] = main_expr.replace('>', '>=') + f' + 1e-8 ' + for_loop
        elif '<' in main_expr and not '<=' in main_expr and not '< =' in main_expr:
            new_dict[key]  = f'1e-8 + ' + main_expr.replace('<', '<=') + ' ' + for_loop
        else: 
            new_dict[key] = value

        if not_contains_inequalities(main_expr):
            new_dict[key] = value
    return new_dict



def is_inequality_ge_zero(expression_str):
    """
    Detects if the inequality string is of the form:
    - expression >= 0
    - 0 <= expression
    Accounts for spaces within operators and different zero representations.
    
    Parameters:
        expression_str (str): The inequality expression as a string.
        
    Returns:
        bool: True if the inequality is expression >= 0 or 0 <= expression, else False.
    """
    # Define zero pattern: '0' or '0.0', '0.00', etc.
    zero_pattern = r'0(?:\.0+)?'
    
    # Define operator patterns with optional spaces
    ge_pattern = r'>\s*='  # '>=' with optional space
    le_pattern = r'<\s*='  # '<=' with optional space
    
    # Compile regex patterns
    # Pattern for 'expression >= 0'
    pattern_ge_zero = re.compile(
        rf"""
        ^                           # Start of string
        \s*                         # Optional leading whitespace
        (?P<expr>.+?)               # Non-greedy match for expression
        \s*                         # Optional whitespace
        {ge_pattern}                # '>=' operator with optional space
        \s*                         # Optional whitespace
        {zero_pattern}              # Zero
        \s*$                        # Optional trailing whitespace and end of string
        """,
        re.VERBOSE
    )
    
    # Pattern for '0 <= expression'
    pattern_zero_le = re.compile(
        rf"""
        ^                           # Start of string
        \s*                         # Optional leading whitespace
        {zero_pattern}              # Zero
        \s*                         # Optional whitespace
        {le_pattern}                # '<=' operator with optional space
        \s*                         # Optional whitespace
        (?P<expr>.+?)               # Non-greedy match for expression
        \s*$                        # Optional trailing whitespace and end of string
        """,
        re.VERBOSE
    )
    
    # Check for 'expression >= 0'
    if pattern_ge_zero.match(expression_str):
        return True
    
    # Check for '0 <= expression'
    if pattern_zero_le.match(expression_str):
        return True
    
    # If neither pattern matches, return False
    return False

def replace_inequalities_equal_zero(this_dict):
    new_dict = {}
    for key, value in this_dict.items():
        main_expr, for_loop = separate_constraint_from_for(value)
        if not is_inequality_ge_zero(main_expr):
            new_dict[key] = value
    return new_dict


def replace_inequalities_not_equal(this_dict):
    new_dict = {}
    for key, value in this_dict.items():
        main_expr, for_loop = separate_constraint_from_for(value)
        if '>' in main_expr and not '>=' in main_expr and not '> =' in main_expr:
            new_dict[key] = main_expr.replace('>', '>=') + f' + 1e-8 ' + for_loop
        elif '<' in main_expr and not '<=' in main_expr and not '< =' in main_expr:
            new_dict[key]  = f'1e-8 + ' + main_expr.replace('<', '<=') + ' ' + for_loop
        else: 
            new_dict[key] = value

        if not_contains_inequalities(main_expr):
            new_dict[key] = value
    return new_dict




def replace_parameters_with_variable_dict(this_dict, this_step):
    final_extracted_dict = {}
    for key, value in this_dict.items():
        final_extracted_dict[key] = replace_parameters_with_variable(value, this_step)

    return final_extracted_dict

def clean_and_parse_dict_parameters(dict_string):
    # Remove comments
    dict_string_cleaned = re.sub(r'#.*', '', dict_string)
    # Evaluate expressions like 1/3 within the dictionary
    # This will replace things like '1/3' with its evaluated result.
    dict_string_cleaned = re.sub(r"(\d+)\s*/\s*(\d+)", lambda m: str(float(m.group(1)) / float(m.group(2))), dict_string_cleaned)
    # Convert the cleaned string to a Python object using ast.literal_eval
    try:
        return ast.literal_eval(dict_string_cleaned)
    except:
        return {}

def extract_dict_from_string(input_string, this_step):
    # Find the start and end of the dictionary
    # dict_start = input_string.find("{")
    # dict_end = input_string.rfind("}") + 1

    # # Extract the dictionary string
    # dict_string = input_string[dict_start:dict_end]
    dict_string = extract_dictionary_content(input_string)

    try:
        # Use ast.literal_eval to safely evaluate the dictionary string
        extracted_dict = ast.literal_eval(dict_string)
        if this_step == 'equality_constraints' or this_step == 'inequality_constraints':
             if not check_none_key_value(extracted_dict):
                extracted_dict = eliminate_list_comprehension_brackets(extracted_dict)
        if this_step == 'equality_constraints':
            if  not check_none_key_value(extracted_dict):
                extracted_dict = fix_single_equals(extracted_dict)
                extracted_dict = replace_inequalities(extracted_dict)
            
        if this_step == 'inequality_constraints':
            if not check_none_key_value(extracted_dict):
                extracted_dict = replace_inequalities_equal_zero(extracted_dict)
                extracted_dict = replace_inequalities_not_equal(extracted_dict)
                if len(extracted_dict) == 0:
                    extracted_dict = {None: None}               
        # if this_step == 'inequality_constraints':
        #         extracted_dict = replace_inequalities_equal_zero(extracted_dict)
        #         extracted_dict = replace_inequalities_not_equal(extracted_dict)
    except:
        if this_step == 'parameters':
            extracted_dict = clean_and_parse_dict_parameters(dict_string)
        else:
            extracted_dict = {}

    if this_step == 'parameters':
        return  dict_string, extracted_dict
    if check_none_key_value(extracted_dict):
        return dict_string, extracted_dict

    try:
        final_extracted_dict = replace_parameters_with_variable_dict(extracted_dict, this_step)
    except:
        final_extracted_dict = {}
    
    return dict_string, final_extracted_dict


def extract_dictionary_content(s):
    """
    Extracts the content of the dictionary assigned to formalization_dict[...] from the given string.

    Parameters:
        s (str): The input string containing the dictionaries.

    Returns:
        dict: A dictionary where keys are the keys inside formalization_dict and values are the extracted dictionary contents as strings.
    """
    # Pattern to find assignments to formalization_dict[...]
    pattern = r"formalization_dict\[(.+?)\]\s*=\s*{"
    matches = re.finditer(pattern, s)
    extracted_dicts = {}

    for match in matches:
        key = match.group(1).strip()
        # Start position of the opening brace
        start = match.end() - 1
        stack = []
        pos = start
        while pos < len(s):
            char = s[pos]
            if char == '{':
                stack.append('{')
            elif char == '}':
                stack.pop()
                if not stack:
                    # Found the matching closing brace
                    end = pos + 1
                    break
            pos += 1
        else:
            # Did not find a matching closing brace
            continue

        # Extract the dictionary content
        dict_content = s[start:end]
        extracted_dicts[key] = dict_content.strip()

    return dict_content.strip()


def parse_dictionaries_from_string(input_string):
    # Regular expression to find the start of the dictionary assignment
    dict_start_regex = r"formalization_dict\['decision_variables'\]\s*=\s*{"
    
    # Find all matches in the input string
    dict_positions = [(m.start(0), m.end(0)) for m in re.finditer(dict_start_regex, input_string)]
    
    dict_strings = []
    evaluated_dicts = []
    
    for start_pos, end_pos in dict_positions:
        brace_count = 1
        end_brace_pos = end_pos
        
        while brace_count > 0 and end_brace_pos < len(input_string):
            if input_string[end_brace_pos] == '{':
                brace_count += 1
            elif input_string[end_brace_pos] == '}':
                brace_count -= 1
            end_brace_pos += 1
        
        # Extract the dictionary string without the prefix
        dict_str = input_string[end_pos - 1:end_brace_pos]  # Adjust to remove the prefix
        dict_strings.append(dict_str)
        
        try:
            # Evaluate the dictionary part
            evaluated_dict = ast.literal_eval(dict_str)
            evaluated_dicts.append(evaluated_dict)
        except Exception as e:
            print(f"Error evaluating dictionary: {e}")
            evaluated_dicts.append(None)
    
    return dict_strings, evaluated_dicts

import ast

def extract_dictionary_group(text, num_objects, step):
    for i in range(num_objects):
        text = text.replace(f"\"{step}_{i}\"", f'{step}_{i}').replace(f"'{step}_{i}'", f'{step}_{i}').replace(f'{step}_{i}', f"'{step}_{i}'")
    # Find the starting point of the dictionary using the keyword "groups"
    text = text.replace('groups =', 'groups=')
    start_index = text.find('groups=')
    
    if start_index == -1:
        return None  # If 'groups =' not found, return None
    
    # Extract the part of the string that contains the dictionary
    dict_start = text.find('{', start_index)
    dict_end = text.find('}', dict_start) + 1
    
    # Extract the dictionary string
    dict_string = text[dict_start:dict_end]
    
    try:
        # Convert the dictionary string to a Python dictionary
        extracted_dict = ast.literal_eval(dict_string)
        return extracted_dict
    except (SyntaxError, ValueError):
        return None  # If the string cannot be parsed as a dictionary, return None

def extract_dictionary_rank(text, num_objects, step):
    for i in range(num_objects):
       #text = text.replace(f"'{step}_{i}'", f'{step}_{i}').replace(f'{step}_{i}', f"'{step}_{i}'")
       text = text.replace(f"\"{step}_{i}\"", f'{step}_{i}').replace(f"'{step}_{i}'", f'{step}_{i}').replace(f'{step}_{i}', f"'{step}_{i}'")
    # Find the starting point of the dictionary using the keyword "groups"
    text = text.replace('Rank', 'rank').replace('rank =', 'rank=')
    start_index = text.find('rank=')
    
    if start_index == -1:
        return None  # If 'groups =' not found, return None
    
    # Extract the part of the string that contains the dictionary
    dict_start = text.find('{', start_index)
    dict_end = text.find('}', dict_start) + 1
    
    # Extract the dictionary string
    dict_string = text[dict_start:dict_end]
    
    try:
        # Convert the dictionary string to a Python dictionary
        extracted_dict = ast.literal_eval(dict_string)
        return extracted_dict
    except (SyntaxError, ValueError):
        return None  # If the string cannot be parsed as a dictionary, return None


def extract_dictionary_rank_final(text, num_objects, step):
    for i in range(1, num_objects + 1):
       #text = text.replace(f"'{step}_{i}'", f'{step}_{i}').replace(f'{step}_{i}', f"'{step}_{i}'")
       text = text.replace(f"\"{step}_{i}\"", f'{step}_{i}').replace(f"'{step}_{i}'", f'{step}_{i}').replace(f'{step}_{i}', f"'{step}_{i}'")
    # Find the starting point of the dictionary using the keyword "groups"
    text = text.replace('Rank', 'rank').replace('rank =', 'rank=')
    start_index = text.find('rank=')
    
    if start_index == -1:
        return None  # If 'groups =' not found, return None
    
    # Extract the part of the string that contains the dictionary
    dict_start = text.find('{', start_index)
    dict_end = text.find('}', dict_start) + 1
    
    # Extract the dictionary string
    dict_string = text[dict_start:dict_end]
    
    try:
        # Convert the dictionary string to a Python dictionary
        extracted_dict = ast.literal_eval(dict_string)
        return extracted_dict
    except (SyntaxError, ValueError):
        return None  # If the string cannot be parsed as a dictionary, return None


def extract_score_final(text):
    # Use regex to find the value of "possible_solution"
    match = re.search(r'"possible_solution":\s*(\d+)', text)
    if match:
        return float(match.group(1))
    else:
        return None

def extract_integers_from_dict(input_dict):
    extracted_integers = []
    for key, value_list in input_dict.items():
        if value_list:  # Check if the list is not empty
            first_element = value_list[0]
            # Extract the integer part from the string
            integer_part = int(first_element.split('_')[-1])
            extracted_integers.append(integer_part)
    return extracted_integers

def extract_unique_group_idx(list_text, num_objects, step):
    list_dict = [extract_dictionary_group(text, num_objects, step) for text in list_text]
    list_dict = [obj for obj in list_dict if obj is not None]
    this_dict = most_common_clustering(list_dict)
    return extract_integers_from_dict(this_dict)

from collections import Counter

def most_common_clustering(list_dicts):

    list_results = [list(this_dict.values()) for this_dict in list_dicts]
    # Normalize the clusters in each result by sorting within clusters and across clusters
    normalized_results = [tuple(sorted(map(tuple, [sorted(cluster) for cluster in result]))) for result in list_results]
    
    # Count the frequency of each clustering decision
    cluster_counts = Counter(normalized_results)
    
    # Get the most common clustering decision
    most_common = cluster_counts.most_common(1)[0][0]
    
    # Convert back to the list structure for easier readability
    most_common_clusters = [list(cluster) for cluster in most_common]
    

    final_results = {}
    for idx, this_result in enumerate(most_common_clusters):
        final_results[idx] = this_result
    return final_results


def update_rank(current_dict, rank):
    for r, variable in rank.items():
        if variable in current_dict:
            current_dict[variable] += r
        else:
            current_dict[variable] = r

    return current_dict

import numpy as np
def get_normalized_rank(dict_1):
    def rank_array(arr):
        descending_order_indices = np.argsort(-arr)
        ranks = np.empty_like(descending_order_indices)
        ranks[descending_order_indices] = np.arange(len(arr))
        return ranks

    # Extract values from the dictionary and convert to numpy array
    dict_values = np.array(list(dict_1.values()))
    this_rank   = rank_array(dict_values)
    
    # Calculate the length of the dictionary and generate the aux_lin array
    aux_lin = np.linspace(0, 1, len(dict_values) + 2)[1:-1]
    return aux_lin[this_rank]


def parse_gurobi_output(text):
    # Regular expressions for capturing relevant information
    best_objective_pattern = r"Best objective\s+([-\d.e+]+),\s+best bound\s+([-\d.e+]+),\s+gap\s+([-\d.e+%]+)"
    root_relaxation_pattern = r"Root relaxation:\s+objective\s+([-\d.e+]+)"
    
    # Prepare result dictionary
    results = {
        'best_objective': None,
        'best_bound': None,
        'gap': None,
        'root_relaxation': None
    }
    
    # Search for "Best objective" information
    best_obj_match = re.search(best_objective_pattern, text)
    if best_obj_match:
        results['best_objective'] = best_obj_match.group(1)
        results['best_bound'] = best_obj_match.group(2)
        results['gap'] = best_obj_match.group(3)
    
    # Search for "Root relaxation" information
    root_relaxation_match = re.search(root_relaxation_pattern, text)
    if root_relaxation_match:
        results['root_relaxation'] = root_relaxation_match.group(1)
    
    return results


import json
def convert_tuple_keys_to_str(d):
    if isinstance(d, dict):
        new_dict = {}
        for key, value in d.items():
            if isinstance(key, tuple):
                key = str(key)  # Convert tuple to string, or use list(key) to convert to a list
            if isinstance(value, dict):
                value = convert_tuple_keys_to_str(value)
            elif isinstance(value, list):
                value = [convert_tuple_keys_to_str(item) if isinstance(item, dict) else item for item in value]
            new_dict[key] = value
        return new_dict
    else:
        return d
    
import ast
import re

def convert_str_keys_to_tuple(d):
    if isinstance(d, dict):
        new_dict = {}
        for key, value in d.items():
            if isinstance(key, str) and key.startswith("(") and key.endswith(")"):
                try:
                    key = ast.literal_eval(key)  # Convert string representation of tuple back to tuple
                except:
                    pass
            if isinstance(value, dict):
                value = convert_str_keys_to_tuple(value)
            elif isinstance(value, list):
                value = [convert_str_keys_to_tuple(item) if isinstance(item, dict) else item for item in value]
            new_dict[key] = value
        return new_dict
    else:
        return d

def parse_gurobi_output(text):
    # Regular expressions for capturing relevant information
    best_objective_pattern = r"Best objective\s+([-\d.e+]+),\s+best bound\s+([-\d.e+]+),\s+gap\s+([-\d.e+%]+)"
    root_relaxation_pattern = r"Root relaxation:\s+objective\s+([-\d.e+]+)"
    optimal_objective_pattern = r"Optimal objective\s+([-\d.e+]+)"
    # Prepare result dictionary
    results = {
        'best_objective': None,
        'root_relaxation': None
    }
    # Search for "Best objective" information
    best_obj_match = re.search(best_objective_pattern, text)
    if best_obj_match:
        results['best_objective'] = best_obj_match.group(1)
    # Search for "Root relaxation" information
    root_relaxation_match = re.search(root_relaxation_pattern, text)
    if root_relaxation_match:
        results['root_relaxation'] = root_relaxation_match.group(1)
    # Search for "Optimal objective" information
    optimal_obj_match = re.search(optimal_objective_pattern, text)
    if optimal_obj_match:
        results['best_objective'] = optimal_obj_match.group(1)
    return results
