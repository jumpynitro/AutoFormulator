import subprocess
import sympy as sp
import os

# Example usage

def create_directories(path):
    os.makedirs(path, exist_ok=True)
    print(f"Directories created or already exist: {path}")

### Differences in parameters
from utils_general import transform_keys

def dict_values_equal(dict1, dict2):
    dict1 = transform_keys(dict1)
    dict2 = transform_keys(dict2)
    def normalize_value(value):
        if isinstance(value, dict):
            return sorted((k, normalize_value(v)) for k, v in value.items())
        elif isinstance(value, list):
            return sorted(normalize_value(v) for v in value)
        return value

    def get_all_values(d):
        all_values = []
        for v in d.values():
            if isinstance(v, dict):
                all_values.extend(get_all_values(v))
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        all_values.extend(get_all_values(item))
                    else:
                        all_values.append(item)
            else:
                all_values.append(v)
        return all_values

    values1 = get_all_values(dict1)
    values2 = get_all_values(dict2)

    normalized_values1 = sorted(normalize_value(v) for v in values1)
    normalized_values2 = sorted(normalize_value(v) for v in values2)

    return normalized_values1 == normalized_values2


def get_dv_sym_str(var_name, space_index = []):
    if len(space_index) == 0:
        return f"{var_name} = sp.symbols('{var_name}')"
    elif len(space_index) == 1:
        return f"{var_name} = {{i: sp.symbols( ('{var_name}_' + str(i)).replace(' ', '_') ) for i in {space_index[0]}}}" 
    elif len(space_index) == 2:
        return f"{var_name} = {{(i,j): sp.symbols(  ('{var_name}_' + str(i) + '_' +  str(j)).replace(' ', '_') ) for i in {space_index[0]} for j in {space_index[1]}}}" 

from utils_general import get_param_code_str, extract_range_dv, get_var_name

def get_symbolic_par_dv(this_dict):

    output_string = "import sympy as sp\n\n"
    
    # Define symbolic variables for parameters
    output_string += "# Define symbolic variables\n"

    output_string += get_param_code_str(this_dict)

    # Define decision variables
    decision_variables = this_dict.get("decision_variables", {})
    output_string += "\n# Decision Variables\n"
    
    for var, description in decision_variables.items():
        #space_index       = extract_range(description)
        space_index       = extract_range_dv(description)
        var_name, is_dict = get_var_name(var)
        this_code_str     = get_dv_sym_str(var_name, space_index)
        output_string    += f'{this_code_str}\n'
    return output_string


### Differences in objectives

def dict_to_obj_str(this_dict):

    # Start with an empty string
    output_string = get_symbolic_par_dv(this_dict)

    # Define the objective functions
    this_idx = 1
    for i, (obj_name, obj_expr) in enumerate(this_dict.items(), start=1):
        if obj_name.startswith('objective_'):
            output_string += f"# Define the objective function {this_idx}\n"
            if 'min' in obj_expr.keys():
                this_obj = obj_expr['min'].replace('\n', '')
            else:
                this_obj = obj_expr['max'].replace('\n', '')

            output_string += f"objective{this_idx} = {this_obj}\n\n"
            this_idx += 1
    
    # Simplify both objective functions
    output_string += "# Simplify both objective functions\n"
    output_string += "simplified_obj1 = sp.simplify(objective1)\n"
    output_string += "simplified_obj2 = sp.simplify(objective2)\n\n"
    
    # Check if they are equivalent
    output_string += "# Check if they are equivalent\n"
    output_string += "equivalence = sp.simplify(simplified_obj1 - simplified_obj2) == 0\n\n"
    output_string += 'print("Hypothesis are equivalent: ", equivalence)\n'
 
    return output_string

EQ_CNST_STRING = """
def are_constraints_equivalent(constraints1, constraints2):
    # Normalize each equation to have the form `Expr = 0`
    normalized_constraints1 = set(sp.simplify(cnst.lhs - cnst.rhs) for cnst in constraints1)
    normalized_constraints2 = set(sp.simplify(cnst.lhs - cnst.rhs) for cnst in constraints2)

    # Directly compare the two sets
    return normalized_constraints1 == normalized_constraints2

# Check if the constraints are equivalent
equivalence = are_constraints_equivalent(equality_constraints_1, equality_constraints_2)

if equivalence:
    print("Hypothesis are equivalent: True")
else:
    print("Hypothesis are equivalent: False")
"""

from utils_general import separate_constraint_from_for, get_borders_constraints, get_prompt_cte_constraints

def dict_to_eq_const_str(this_dict):
    # Initialize the output string
    output_string = get_symbolic_par_dv(this_dict)

    for aux_const_dict in ["equality_constraints_1", "equality_constraints_2"]:
        borders_constraints  = get_borders_constraints(this_dict[aux_const_dict], check_is_constant = True,
                                                        parameter_list = list(this_dict['parameters'].keys())  )
        output_string       += get_prompt_cte_constraints(borders_constraints, this_dict, this_dict[aux_const_dict], var_type = 'simpy')

    for key in ["equality_constraints_1", "equality_constraints_2"]:
        constraints = this_dict.get(key, {})
        output_string += f"\n# {key.replace('_', ' ').capitalize()}\n"
        
        output_string += f'{key} = []\n'
        for cons_name, const_str in constraints.items():
            eq_const, for_loop = separate_constraint_from_for(const_str)
            output_string += f"{key} += [sp.Eq({eq_const.split('==')[0].strip()}, {eq_const.split('==')[1].strip()}) {for_loop}]\n"

    # Add the function to check for constraint equivalence
    output_string += EQ_CNST_STRING
    
    return output_string

def split_by_inequality(expression):
    # Updated regex pattern that accounts for possible spaces between operators
    pattern = r'\s*(<=|>=|<\s*=|>\s*=|[<>!=]=?|==|=)\s*'
    # Use regex to split the expression
    split_parts = re.split(pattern, expression, maxsplit=1)
    # The operator will be at index 1, and the expressions on either side will be at index 0 and 2
    return split_parts[0].strip(), split_parts[2].strip()


def dict_to_ineq_const_str(this_dict):
    # Initialize the output string
    output_string = get_symbolic_par_dv(this_dict)

    # Define equality constraints
    for key in ["inequality_constraints_1", "inequality_constraints_2"]:
        constraints = this_dict.get(key, {})
        output_string += f"\n# {key.replace('_', ' ').capitalize()}\n"
        output_string += f'{key} = []\n'
        for cons_name, const_str in constraints.items():
            ineq_const, for_loop = separate_constraint_from_for(const_str)
            lhs, rhs = split_by_inequality(ineq_const)
            if '<=' in ineq_const or '< =' in ineq_const or '<' in ineq_const:
                output_string += f"{key} += [sp.Eq({lhs}, {rhs}) {for_loop}]\n"
            else:
                output_string += f"{key} += [sp.Eq({rhs}, {lhs}) {for_loop}]\n"

    # Add the function to check for constraint equivalence
    aux1 = EQ_CNST_STRING.replace("equality_constraints_1", "inequality_constraints_1")
    aux1 = aux1.replace("equality_constraints_2", "inequality_constraints_2")
    output_string += aux1
    
    return output_string

import re
def extract_equivalence_value(input_string):
    # Define the pattern to match the string format without angle brackets
    pattern = r"Hypothesis are equivalent:\s*(True|False)"
    
    # Search for the pattern in the input string
    match = re.search(pattern, input_string)
    
    # If a match is found, return the boolean equivalent of the matched value
    if match:
        value = match.group(1)
        if value == 'True':
            return True
        elif value == 'False':
            return False
    
    # If no match is found or value is not 'True' or 'False', return False
    return False


def execute_python_code(code_str, python_filename = "temp_script.py", output_filename="output.txt"):
    # Write the Python code to a temporary file
    with open(python_filename, "w") as script_file:
        script_file.write(code_str)

    # Execute the script and redirect output to a text file
    with open(output_filename, "w") as output_file:
        subprocess.run(["python3", python_filename], stdout=output_file, stderr=subprocess.STDOUT)

    # Read the output from the text file into a variable
    with open(output_filename, "r") as output_file:
        output_content = output_file.read()

    output_content = extract_equivalence_value(output_content)

    return output_content


def filter_functionally_equivalent(original_objects, path):
    objects          = [obj for obj in original_objects if obj.run_code_if_code]
    aux_objects      = [obj for obj in original_objects if not obj.run_code_if_code]

    if len(aux_objects) > 1:
        add_unique_objects    = [aux_objects[0]]
        add_discarded_objects = aux_objects[1:]  
    else:
        add_unique_objects    = aux_objects
        add_discarded_objects = []

    unique_objects = []
    discarded_objects = []

    while objects:
        current_object = objects.pop(0)
        # Add the current object to unique objects list
        unique_objects.append(current_object)

        # Iterate over the remaining objects and compare with current object
        non_equivalent_objects = []
        for obj in objects:

            objects_are_equivalent = get_equivalence_function(current_object, obj, path)
            if not objects_are_equivalent:
                non_equivalent_objects.append(obj)
            else:
                # If equivalent, add to discarded objects
                discarded_objects.append(obj)

        # Update the list with non-equivalent objects only
        objects = non_equivalent_objects

    unique_objects    += add_unique_objects
    discarded_objects += add_discarded_objects
    return unique_objects, discarded_objects

def rename_keys(d, pre_str, post_str):
    return {k.replace(pre_str, post_str): v for k, v in d.items()}

def obtain_code_compar(dict1, dict2, step, is_symbolic = False):
    dict_aux, dict2_aux = rename_keys(dict1, step, step + '_1'), rename_keys(dict2, step, step + '_2')
    dict_aux.update(dict2_aux)
    if step == 'objective':
        code_str     = dict_to_obj_str(dict_aux)
    elif step == 'equality_constraints':
        if is_symbolic:
            code_str     = dict_to_eq_const_str(dict_aux)
        else:
            code_str     = dict_to_eq_const_str_smt(dict_aux)
    elif step == 'inequality_constraints':
        if is_symbolic:
            code_str     = dict_to_ineq_const_str(dict_aux)
        else:
            code_str     = dict_to_ineq_const_str_smt(dict_aux)
    return code_str

def compar_two_dicts(dict1, dict2, idx1, idx2, path, step, is_symbolic = False):
    if step == 'parameters':
        is_different = dict_values_equal(dict1['parameters'], dict2['parameters'])
    elif step in ['objective', 'equality_constraints', 'inequality_constraints']:
        code_str      = obtain_code_compar(dict1, dict2, step, is_symbolic = is_symbolic)
        is_different  = execute_python_code(code_str, f"{path}/code-aux-{idx1}-{idx2}.py", f"{path}/output-{idx1}-{idx2}.txt")
    return is_different

def get_equivalence_function(obj1, obj2, path):
    step  = obj1.this_step
    dict1 = obj1.form_dict_eval.copy()
    dict2 = obj2.form_dict_eval.copy()
    return compar_two_dicts(dict1, dict2, obj1.idx, obj2.idx, path, step)



def filter_parameters_before(objects):
    if not objects:
        return [], []
    
    # Initialize variables
    major_list = []
    discarded_list = []
    
    # Identify the object with the largest 'parameters' dictionary
    max_length = 0
    major_object = None
    
    for obj in objects:
        # Get the length of the 'parameters' dictionary
        dict_length = len(obj.form_dict_eval.get('parameters', {}))
        
        # Update major_object if a larger dictionary is found
        if dict_length > max_length:
            max_length = dict_length
            major_object = obj
    
    # Populate the major_list and discarded_list
    major_list.append(major_object)
    discarded_list = [obj for obj in objects if obj != major_object]
    
    return major_list, discarded_list

def get_most_common_dict_index(dict_list):
    from collections import defaultdict

    def make_hashable(o):
        if isinstance(o, (list, tuple)):
            return tuple(make_hashable(e) for e in o)
        elif isinstance(o, dict):
            return tuple(sorted((make_hashable(k), make_hashable(v)) for k, v in o.items()))
        else:
            return o

    hashable_dicts = []
    index_map = defaultdict(list)
    count_map = defaultdict(int)

    for index, d in enumerate(dict_list):
        # Convert the dictionary to a hashable representation
        hashable = make_hashable(d)
        hashable_dicts.append(hashable)
        count_map[hashable] += 1
        index_map[hashable].append(index)

    max_count = max(count_map.values())
    max_count_hashables = [h for h, c in count_map.items() if c == max_count]

    if max_count > 1:
        # There is at least one dictionary that repeats
        # Pick any one of the dictionaries that repeats the most
        # Return any one index of that dictionary
        selected_hashable = max_count_hashables[0]
        selected_index = index_map[selected_hashable][0]
        return selected_index
    else:
        # No repetitions, pick the dictionary with the highest number of elements
        max_elements = -1
        selected_index = -1
        for index, d in enumerate(dict_list):
            num_elements = len(d)
            if num_elements > max_elements:
                max_elements = num_elements
                selected_index = index
        return selected_index

def filter_by_mode(objects):
    if not objects:
        return [], []
    dict_list = []
    for obj in objects:
        dict_list += [obj.form_dict_eval.get('parameters', {})]
    idx_selected = get_most_common_dict_index(dict_list)
    best_object  = objects[idx_selected]
    discarded_list = [obj for obj in objects if obj != best_object]
    return [best_object], discarded_list
        
def filter_parameters(objects):
    try:
        unique_list, discarded_list = filter_by_mode(objects)
    except:
        unique_list, discarded_list = filter_by_mode(objects)
    return unique_list, discarded_list





EQ_CNST_STRING_SMT = """
def check_equivalence_smt(system1, system2, variables):
    # input two system of linear equations

    eq_flag = True
    s = Solver()
    
    s.push()
    # check that solution to both equations in system 1 is not a solution to either equation in system 2
    s.add(And(*[eq for eq in system1]))
    s.add(Or(*[Not(eq) for eq in system2]))
    # check satisfiability
    result = s.check()
    if result == sat:
        print("Hypothesis are equivalent: False")
        return
        eq_flag=False
    s.pop()

    s.push()
    # reverse check: solution to both equations in system 2 is not a solution to either of system 1
    s.add(And(*[eq for eq in system2]))
    s.add(Or(*[Not(eq) for eq in system1]))
    result = s.check()
    if result == sat:
        print("Hypothesis are equivalent: False")
        return
        eq_flag=False
    s.pop()

    if eq_flag:
        print("Hypothesis are equivalent: True")
        return
    return eq_flag

check_equivalence_smt(system1, system2, all_variables)
"""



def get_dv_smt_str(output_string, var_name, space_index = []):
    if len(space_index) == 0:
        output_string += f"{var_name} = Reals('{var_name}')[0]\n"
        output_string += f"all_variables += [{var_name}]\n"
    elif len(space_index) == 1:
        output_string += f"{var_name} = {{i: Reals( ('{var_name}_' + str(i)).replace(' ', '_') )[0] for i in {space_index[0]}}}\n"
        output_string += f"all_variables += [Reals( ('{var_name}_' + str(i)).replace(' ', '_') )[0] for i in {space_index[0]}]\n"
    elif len(space_index) == 2:
        output_string += f"{var_name} = {{(i,j): Reals(  ('{var_name}_' + str(i) + '_' +  str(j)).replace(' ', '_') )[0] for i in {space_index[0]} for j in {space_index[1]}}}\n" 
        output_string += f"all_variables += [Reals(  ('{var_name}_' + str(i) + '_' +  str(j)).replace(' ', '_') )[0]  for i in {space_index[0]} for j in {space_index[1]}]\n"
    return output_string

from utils_general import get_param_code_str, extract_range_dv, get_var_name

def get_smt_par_dv(this_dict):

    output_string = "from z3 import Solver, Reals, And, Or, sat, Not\n\n"
    
    # Define symbolic variables for parameters
    output_string += "# Define symbolic variables\n"

    output_string += get_param_code_str(this_dict)

    # Define decision variables
    decision_variables = this_dict.get("decision_variables", {})
    output_string += "\n# Decision Variables\n"
    
    output_string += "all_variables = []\n"
    for var, description in decision_variables.items():
        #space_index       = extract_range(description)
        space_index       = extract_range_dv(description)
        var_name, is_dict = get_var_name(var)
        output_string     = get_dv_smt_str(output_string, var_name, space_index)
        #output_string    += f'{this_code_str}\n'
    return output_string


def dict_to_eq_const_str_smt(this_dict):
    # Initialize the output string
    output_string = get_smt_par_dv(this_dict)

    for aux_const_dict in ["equality_constraints_1", "equality_constraints_2"]:
        borders_constraints  = get_borders_constraints(this_dict[aux_const_dict], check_is_constant = True,
                                                       parameter_list = list(this_dict['parameters'].keys()) )
        output_string       += get_prompt_cte_constraints(borders_constraints, this_dict, this_dict[aux_const_dict], var_type = 'smt')

    dict_name = {'equality_constraints_1': 'system1', 'equality_constraints_2': 'system2'}
    for key in ["equality_constraints_1", "equality_constraints_2"]:
        constraints = this_dict.get(key, {})
        output_string += f"\n# {key.replace('_', ' ').capitalize()}\n"
        
        output_string += f'{dict_name[key]} = []\n'
        for cons_name, const_str in constraints.items():
            #eq_const, for_loop = separate_constraint_from_for(const_str)
            output_string += f"{dict_name[key]} += [{const_str}]\n"

    # Add the function to check for constraint equivalence
    output_string += EQ_CNST_STRING_SMT
    return output_string



def dict_to_ineq_const_str_smt(this_dict):
    # Initialize the output string
    output_string = get_smt_par_dv(this_dict)

    dict_name = {'inequality_constraints_1': 'system1', 'inequality_constraints_2': 'system2'}
    for key in ["inequality_constraints_1", "inequality_constraints_2"]:
        constraints = this_dict.get(key, {})
        output_string += f"\n# {key.replace('_', ' ').capitalize()}\n"
        
        output_string += f'{dict_name[key]} = []\n'
        for cons_name, const_str in constraints.items():
            #eq_const, for_loop = separate_constraint_from_for(const_str)
            output_string += f"{dict_name[key]} += [{const_str}]\n"

    # Add the function to check for constraint equivalence
    output_string += EQ_CNST_STRING_SMT
    return output_string