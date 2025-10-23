import re
import ast
import subprocess

def check_none_key_value(d):
    return d.get(None) is None and None in d

def transform_keys(data):
    for key, value in data.items():
        if isinstance(value, dict):  # Check if the value is a dictionary
            new_dict = {}
            for sub_key, sub_value in value.items():
                try:
                    new_key = float(sub_key)  # Try to convert the key to an integer
                except:
                    new_key = sub_key  # Keep the key as a string if it can't be converted
                new_dict[new_key] = sub_value
            data[key] = new_dict  # Replace the old dictionary with the new one
    return data

def get_param_code_str(formalization_dict, apply_ast = False):
    if apply_ast:
        param_dict = ast.literal_eval(formalization_dict['parameters'])
    else:
        param_dict = formalization_dict['parameters']
    param_dict = transform_keys(param_dict)
    prompt = ""
    for param, value in param_dict.items():
        if type(value) == str:
            prompt += f'{param} = "{str(value)}"\n'
        else:
            prompt += f'{param} = {value}\n'
    return prompt


def extract_vtype(input_string):
    # Use regex to find the text between < and >
    match = re.search(r'<(.*?)>', input_string)
    if match:
        return match.group(1)
    else:
        return None

def extract_vtype_dv(input_dict):
    # Use regex to find the text between < and >
    return input_dict['type']

def get_var_name(this_var):
    right_index = -1
    if this_var[right_index] == ']':
        left_index = this_var[:right_index].rfind("[")
        return this_var[:left_index], True
    else:
        return this_var, False
        
def extract_range(input_string):
    # Regular expression to match content inside {}
    return re.findall(r'\{(.*?)\}', input_string)

def extract_range_iterables(comp_str):
    if comp_str is None:
        return ''
    try:
        # Parse the string as a Python expression
        comp_ast = ast.parse(comp_str, mode='eval')
        # Check if the parsed expression is a List Comprehension
        if isinstance(comp_ast.body, ast.ListComp):
            iterables = []
            for gen in comp_ast.body.generators:
                iterable = gen.iter
                # Use ast.unparse to convert the AST node back to a string
                iter_str = ast.unparse(iterable)
                iterables.append(iter_str)
            return iterables
    except Exception as e:
        print(f"Error parsing comprehension: {e}")
        return ""
    return []



def extract_range_dv(input_dict):
    #def extract_iterables(comp_str):
    comp_str = input_dict['iteration_space']
    if comp_str is None:
        return ''
    
    matches = extract_range_iterables(comp_str)
    return matches

def extract_variable_name(equation):
    # Match everything before the first opening bracket
    match = re.match(r"([a-zA-Z_]+)\[", equation)
    if match:
        return match.group(1)
    return None

def remove_brackets_content(input_string):
    # Use regex to remove the brackets and their content
    result = re.sub(r"\[.*?\]", "", input_string)
    return result.strip()  # Remove any extra spaces


def separate_constraint_from_for(input_string):
    # Remove leading/trailing whitespace
    input_string = input_string.strip()
    stack = []
    for_loop_starts = []
    for i, char in enumerate(input_string):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
        elif input_string[i:i+4] == ' for' and not stack:
            for_loop_starts.append(i)
    if for_loop_starts:
        main_expr = input_string[:for_loop_starts[0]].strip()
        for_loop = input_string[for_loop_starts[0]:].strip()
    else:
        main_expr = input_string
        for_loop = ""
    return main_expr, for_loop


def extract_loop_variables(loop_string):
    # Remove any content after an "if" statement (for filtering conditions)
    loop_string = loop_string.split(' if ')[0]    
    # Find all variables that appear right after the 'for' keyword and before 'in'
    variables = re.findall(r'for\s+([a-zA-Z_]\w*)|for\s+\((.*?)\)\s+in', loop_string)
    # Extract the first group of results from tuples in the findall result
    results = [v for group in variables for v in group if v]
    # Split tuples inside the loop and flatten the list
    flattened_results = []
    for item in results:
        flattened_results.extend(item.split(', '))
    return flattened_results


def is_constant_constraint(equation: str, allowed_variables: list):
    # Step 1: Eliminate the for loop if it exists at the end of the string
    equation = re.sub(r'\s+for\s+\w+\s+in\s+\w+', '', equation).strip()
    
    # Step 2: Split the equation into left and right parts by "=="
    if "==" not in equation:
        return False, None  # Not a valid equality
    left, right = map(str.strip, equation.split("==", 1))
    
    # Step 3: Check the right side (it should be a number or a single variable, with or without brackets)
    # Regex pattern for a number (integer or float)
    if re.match(r'^[-+]?\d*\.?\d+$', right):
        pass  # It's a number, proceed further
    else:
        # Check if the right part is a variable, with or without brackets
        # Match variables with brackets: variable_name[...]
        match_right_with_brackets = re.match(r'^([a-zA-Z_]\w*)\[(.*?)\]$', right)
        if match_right_with_brackets:
            variable_name_right = match_right_with_brackets.groups()[0]
        else:
            variable_name_right = right  # In case there are no brackets, it's the variable itself
        
        # Check if the variable name (without brackets) is in the allowed list
        if variable_name_right not in allowed_variables:
            return False, None  # Variable not in the allowed list
    
    # Step 4: Check the left side (it should be a variable with brackets and at least one index must be an integer)
    match_left = re.match(r'^([a-zA-Z_]\w*)\[(.*?)\]$', left)
    if not match_left:
        return False, None  # Left side must have brackets
    variable_name, index_content = match_left.groups()
    
    # Step 5: Check if at least one index is an integer
    indices = [idx.strip() for idx in index_content.split(',')]
    has_integer_index = any(re.match(r'^\d+$', idx) for idx in indices)
    if not has_integer_index:
        return False, None
    
    # If all conditions are satisfied, return True and the list of indices
    return True, indices


def get_borders_constraints(eq_const_dict, check_is_constant = False, parameter_list = []):
    all_borders_constraints = []
    for eq_const in eq_const_dict.keys():
        if check_is_constant:
            try:
                is_constant, indexes = is_constant_constraint(eq_const_dict[eq_const], parameter_list)
            except:
                import pdb
                pdb.set_trace()
                print("A")
            if is_constant:
                all_borders_constraints += [(eq_const, indexes)]
    return all_borders_constraints

def create_dict_str(this_dict):
    prompt = "final_dict = {'space': {}, 'vtype': {}}\n"
    for dv_name, dv_value in this_dict.items():
        #var_name, this_space, vtype = remove_brackets_content(dv_name), extract_range(dv_value), extract_vtype(dv_value)
        var_name, this_space, vtype = remove_brackets_content(dv_name), extract_range_dv(dv_value), extract_vtype_dv(dv_value)
        prompt += f"final_dict['space']['{var_name}'] = [{', '.join(this_space)}]\n"
        prompt += f"final_dict['vtype']['{var_name}'] = '{vtype}'\n"
    return prompt

from collections.abc import KeysView

def get_this_list_values(code_string):
    local_vars = {}
    exec(code_string, {}, local_vars)
    this_list = local_vars.get('final_dict', None)
    if isinstance(this_list, KeysView):
        this_list = list(this_list)
    return this_list

def check_if_int(item):
    try:
        int(item)  # Try to convert the item to an integer
        return True
    except:
        return False

def get_prompt_cte_constraints(all_const_cte, formalization_dict, const_dict, var_type = 'gurobipy'):
    prompt_for_dv   = ""
    prompt_for_dv  += get_param_code_str(formalization_dict)
    prompt_for_dv  += "\n"
    prompt_for_dv  += create_dict_str(formalization_dict['decision_variables'])
    dv_dict         = get_this_list_values(prompt_for_dv)

    def detect_if_update(space_list, indexes):
        for this_space, this_index in zip(space_list, indexes):
            if check_if_int(this_index):
                #print(this_index)
                #print(this_space)
                if not (int(this_index) in this_space):
                    return True
        return False
    
    is_detected = False
    prompt = ""
    for const_name, indexes in all_const_cte:
        var_name      = extract_variable_name(const_dict[const_name])
        space_list    = dv_dict['space'][var_name]
        if detect_if_update(space_list, indexes):
            is_detected = True
            _, for_loop = separate_constraint_from_for(const_dict[const_name])
            vtype       = dv_dict['vtype'][var_name]
            idx_str_1   = ', '.join(indexes)
            if var_type == 'gurobipy':
                idx_str     = ', '.join(f'{{{elem}}}' for elem in indexes)
                prompt     += "%s.update({(%s):  gpy_model.addVar(vtype=%s, name = f'%s[%s]') %s})\n" % \
                                            (var_name, idx_str_1, vtype, var_name, idx_str, for_loop)
            elif var_type == 'simpy':
                idx_str     = '_'.join(f'{{{elem}}}' for elem in indexes)
                prompt     += "%s.update({(%s): sp.symbols(f'%s_%s'.replace(' ', '_') ) %s})\n" % \
                            (var_name, idx_str_1, var_name, idx_str, for_loop)
                
            elif var_type == 'smt':
                idx_str     = '_'.join(f'{{{elem}}}' for elem in indexes)
                prompt     += "%s.update({(%s): Reals(f'%s_%s'.replace(' ', '_') ) %s})\n" % \
                            (var_name, idx_str_1, var_name, idx_str, for_loop)
                prompt     += "all_variables += [Reals(f'%s_%s'.replace(' ', '_') ) %s])\n" % \
                            (var_name, idx_str_1, var_name, idx_str, for_loop)
    if is_detected:
        prompt += "\n\n"
    return prompt
