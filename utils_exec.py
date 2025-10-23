import re
import ast
import subprocess
import numpy as np

PROMPT_IMPORT = """
from gurobipy import Model, GRB

# Initialize the model
gpy_model = Model("Production Planning")

"""

def check_script_output(file_path):
    """
    Checks the contents of output.txt for common Python errors.

    Parameters:
    file_path (str): The path to the output.txt file.

    Returns:
    str: A message indicating whether the script ran correctly or if errors were found.
    """
    # Define common error keywords to search for
    error_keywords = ["Error", "Exception", "Traceback", "fatal", "failed"]
    
    # Read the contents of the output file
    with open(file_path, 'r') as file:
        output_content = file.read()
    
    # Check if any error keywords are in the output content
    for keyword in error_keywords:
        if keyword in output_content:
            return False, f"Errors detected in {file_path}. Check the log for details."
    
    return True, f"No errors detected in {file_path}. Script ran successfully."


def execute_python_code_vanilla(code_str, python_filename = "temp_script.py", output_filename="output.txt"):
    # Write the Python code to a temporary file
    with open(python_filename, "w") as script_file:
        script_file.write(code_str)

    # Execute the script and redirect output to a text file
    with open(output_filename, "w") as output_file:
        subprocess.run(["python3", python_filename], stdout=output_file, stderr=subprocess.STDOUT)

    # Read the output from the text file into a variable
    with open(output_filename, "r") as output_file:
        output_content = output_file.read()
    return output_content

import re
from utils_general import extract_vtype_dv, extract_range_dv, get_var_name


def get_var_code_str(var_name, vtype, space_index = []):
    if len(space_index) == 0:
        return "%s = gpy_model.addVar(vtype=%s, name='%s')" % (var_name, vtype, var_name)
    elif len(space_index) == 1:
        return "%s = {i: gpy_model.addVar(vtype=%s, name=f'%s[{i}]') for i in %s}" % \
                                            (var_name, vtype, var_name, space_index[0])
    elif len(space_index) == 2:
        return "%s = {(i, j): gpy_model.addVar(vtype=%s, name=f'%s[{i},{j}]') for i in %s for j in %s}" % \
                                            (var_name, vtype, var_name, \
                                             space_index[0], space_index[1])
    
def get_var_code_str_print(var_name, vtype, space_index = []):
    if len(space_index) == 0:
        return "%s = gpy_model.addVar(vtype=%s, name='%s')" % (var_name, vtype, var_name)
    elif len(space_index) == 1:
        return "%s = {i: gpy_model.addVar(vtype=%s, name=f'%s[{i}]') for i in %s}" % \
                                            (var_name, vtype, var_name, space_index[0])
    elif len(space_index) == 2:
        return "%s = {(i, j): gpy_model.addVar(vtype=%s, name=f'%s[{i},{j}]') for i in %s for j in %s}" % \
                                            (var_name, vtype, var_name, \
                                             space_index[0], space_index[1])

def get_dv_code_str(formalization_dict, apply_ast = False):
    if apply_ast:
        dv_dict = ast.literal_eval(formalization_dict['decision_variables'])
    else:
        dv_dict = formalization_dict['decision_variables']
    code_prompt = ''
    for this_var in dv_dict.keys():
        #space_index = extract_range_expression(dv_dict[this_var])
        space_index       = extract_range_dv(dv_dict[this_var])
        vtype             = extract_vtype_dv(dv_dict[this_var])
        var_name, is_dict = get_var_name(this_var)
        this_code_str     = get_var_code_str(var_name, vtype, space_index)
        code_prompt      += f'{this_code_str}\n'
    return code_prompt

import re
def clean_python_expression(expression):
    # Remove leading newlines and spaces
    expression = expression.lstrip()
    # Remove comments and empty lines before the actual code
    lines = expression.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('#') or not stripped_line:
            continue  # Skip comment lines and empty lines
        cleaned_lines.append(line)
    # Rejoin the cleaned lines
    cleaned_expression = "\n".join(cleaned_lines)
    return cleaned_expression

def get_obj_code_str(formalization_dict, apply_ast = False, max_or_min_bool = True):
    if apply_ast:
        obj_dict   = ast.literal_eval(formalization_dict['objective'])
    else:
        obj_dict   = formalization_dict['objective']

    max_or_min = list(obj_dict.keys())[0]
    expression = obj_dict[max_or_min]
    expression = clean_python_expression(expression)
    expression = expression.replace('\n', '')
    prompt  = f"objective = {expression}\n"
    if max_or_min_bool:
        if max_or_min == 'min':
            prompt += "gpy_model.setObjective(objective, GRB.MINIMIZE)\n"
        else:
            prompt += "gpy_model.setObjective(objective, GRB.MAXIMIZE)\n"
        prompt += "gpy_model.optimize()"
    return prompt

from  utils_general import separate_constraint_from_for, extract_loop_variables, get_borders_constraints, get_prompt_cte_constraints

def get_prompt_all_constraints(const_dict):
    prompt_aux = ""
    for const_name, const_str in const_dict.items():
        main_constraint, for_loop = separate_constraint_from_for(const_str)
        if for_loop == '':
            prompt_aux    += "all_const += [gpy_model.addConstr(%s, name = '%s')] \n" % (main_constraint, const_name)
        else:
            loop_variables = extract_loop_variables(for_loop)
            loop_index_str = ', '.join(f'{{{elem}}}' for elem in loop_variables)
            prompt_aux    += "all_const += [gpy_model.addConstr(%s, name = f'%s[%s]')  %s] \n" % \
                                                    (main_constraint, const_name, loop_index_str, for_loop)
    return prompt_aux

from utils_general import check_none_key_value

def get_code_for_const_str(formalization_dict, is_equality_const = True, apply_ast = False):
    form_str = 'equality_constraints' if is_equality_const else 'inequality_constraints'
    const_dict = ast.literal_eval(formalization_dict[form_str]) if apply_ast else formalization_dict[form_str]
    if check_none_key_value(const_dict):
        return ""
    prompt                               = ""
    borders_constraints                  = get_borders_constraints(const_dict, check_is_constant = is_equality_const,
                                                    parameter_list = list(formalization_dict['parameters'].keys()) )
    prompt                              += get_prompt_cte_constraints(borders_constraints, formalization_dict, const_dict)
    prompt                              += get_prompt_all_constraints(const_dict)
    return prompt

from utils_general import get_param_code_str

def create_code_str(loaded_dict, apply_ast = False, variables_included = 'parameters'):
    prompt  = PROMPT_IMPORT
    prompt += get_param_code_str(loaded_dict, apply_ast = apply_ast) + "\n\n"
    if variables_included == 'parameters':
        return prompt
    prompt += get_dv_code_str(loaded_dict, apply_ast = apply_ast) + "\n\n"
    if variables_included == 'objective':
        prompt += get_obj_code_str(loaded_dict, apply_ast = apply_ast, max_or_min_bool = False) + "\n\n"
        return prompt
    prompt += "all_const = [] \n\n"
    prompt += get_code_for_const_str(loaded_dict, is_equality_const = True, apply_ast = apply_ast) + "\n\n"
    if variables_included == 'equality_constraints':
        return prompt
    prompt += get_code_for_const_str(loaded_dict, is_equality_const = False, apply_ast = apply_ast) + "\n\n"
    prompt += get_obj_code_str(loaded_dict, apply_ast = apply_ast) + "\n\n"
    return prompt


