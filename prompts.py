TEMPLATE_PROBLEM = \
"""
I have a problem in operational research:
  
------ 
###PROBLEM DESCRIPTION###
------

I have the following formalization:

formalization_dict = {
'parameters': {},
'decision_variables': {},
'objective': {},
'equality_constraints': {},
'inequality_constraints': {},
}
"""
INST_PARAMETERS = \
"You are an optimization modeling expert." \
" Complete formalization_dict based on the problem description," \
" you should complete the 'parameters' field, which consists of assigning constants to descriptive variable names."\
""" Only complete 'parameters' and nothing else. Follow these guidelines:

1. Your primary responsibility is to define all the parameters from the problem description that will later be used to define decision variables, the objective, and constraints (both equality and inequality).
2. You may include additional parameters in a format suitable for facilitating the subsequent tasks of defining decision variables, the objective function, and constraints.
3. For parameters that involve multiple indices (e.g., x[i] or x[i,j]), use the most appropriate data structure, such as lists, dictionaries, or dictionaries with tuple keys, to represent them.
4. For each parameter, include a clear, descriptive comment explaining its meaning.
5. Ensure that the parameter names (keys) are descriptive and intuitive.

Return only the python dictionary update (i.e.,  formalization_dict['parameters'] = ...) following the described requirements.
"""

INST_DV = \
"You are an optimization modeling expert."\
" Complete only the 'decision_variables' field within the 'formalization_dict' based on the provided problem description."\
""" Ensure the decision_variables comprehensively cover all essential elements to accurately model the optimization problem.

Each key-value pair in the dictionary must adhere to the following structure:

<key>: {
    'description': <description>,
    'type': <type>,
    'iteration_space': <space>
}

The structure should meet these requirements:
1. Each <key> represents a decision variable that will later be used to implement the objective, equality, and inequality constraints in a Python program.
2. Replace <key> with a symbolic name representing the decision variable. Ensure that each <key> represents a distinct decision variable with a unique symbolic name.
3. Replace <description> with a detailed explanation of the role of the decision variable in the optimization model.
4. Replace <type> with a string representing the Gurobi variable type (e.g., GRB.INTEGER), as this will be used to create the variable via Gurobi's addVar function.
5. If the decision variable is indexed, replace <space> with a string representing Python for-loop using list comprehension syntax to represent the index space. For this, assume direct access to these parameter variables (i.e., avoid using parameters[variables] syntax).
6. If the variable is not indexed, set <space> to None. 
7. If the variable is indexed, do not write the index in the symbol (do not put the index when writting <key>).
8. You are encouraged to create decision variables that are general. If two decision variables represent the same concept write them as on key, creating an appropriate iteration space.

Return only the Python dictionary update (i.e.,  formalization_dict['decision_variables'] = {...}) following the described requirements.
"""


INST_OBJ = \
"You are an optimization modeling expert."\
" Complete only the 'objective' field within the 'formalization_dict' based on the provided problem description." \
""" Do not complete any other fields. Follow these requirements:

1) Write the objective function mathematically using decision variables.
2) Preface the key-value pair with a Python comment explaining the rationale behind the objective. DO NOT make a commentary inside the mathematical description. 
3) Use parameter-defined variables instead of hard-coded values. Assume direct access to these parameter variables (i.e., avoid using parameters[variables] syntax).
4) The dictionary key must be 'min' or 'max', reflecting the nature of the objective (minimization or maximization).
5) The dictionary value must be a string representation of the objective function based on the problem description, written in valid Python syntax.

Return only the Python dictionary update (i.e.,  formalization_dict['objective'] = {"max": ...} or formalization_dict['objective'] = {"min": ...}) following the described requirements.
"""

INST_EQ_CONST = \
"You are an optimization modeling expert." \
" Complete the formalization_dict by filling in the equality_constraints field based on the problem description and the decision variables provided."\
" These constraints include border constraints, initialization, and equality constraints derived from the problem description."\
""" Do not complete the "inequality_constraints" field. Follow these requirements:

1. Descriptive constraints: Each key in the dictionary should represent a unique, clearly named constraint, with the value being a string that describes the corresponding mathematical equality using "==".
2. Parameter Variables: Use parameter-defined variables instead of hard-coded values. Assume direct access to these parameter variables (i.e., avoid using parameters[variables] syntax).
3. Indexed Variables: For indexed decision variables, indicate the index within brackets (e.g., x[i]).
4. Handling Multiple Constraints: For similar constraints that repeat across indices or variables, use Python for loops and list comprehensions for efficient representation.
5. String mathematical description: Note, the value (mathematical description) should be a single string. DO NOT use .join() or anything else. Even if it represents multiple constraints using a for loop.
6. No Inequality Constraints: Only define equality constraints. Inequality constraints will be handled separately by a subsequent expert. 
7. Comments: Include a Python comment before each key-value pair, explaining the rationale behind the constraint.

Return only the Python dictionary update (i.e.,  formalization_dict['equality_constraints'] = {...}) following these requirements.
Important: If the problem contains only inequality constraints and no equality constraints, return: formalization_dict['equality_constraints'] = {None: None}. This will signal the need to focus on inequality constraints in subsequent modeling steps.
"""


INST_INEQ_CONST = \
"You are an optimization modeling expert." \
" Complete the formalization_dict by adding the inequality_constraints field based on the problem description." \
""" Follow these requirements:

1. Descriptive constraints: Each key in the dictionary should represent a unique, clearly named constraint, with the value being a string that describes the corresponding mathematical inequality.
2. Parameter Variables: Use parameter-defined variables instead of hard-coded values. Assume direct access to these parameter variables (i.e., avoid using parameters[variables] syntax).
3. Indexed Variables: For indexed decision variables, indicate the index within brackets (e.g., x[i]).
4. Handling Multiple Constraints: For similar constraints that repeat across indices or variables, use Python for loops and list comprehensions for efficient representation.
5. String mathematical description: Note, the value (mathematical description) should be a single string without using join or anything else. Even if it represents multiple constraints using a for loop.
6. Inequality Constraints Only: Include only inequality constraints. Exclude any constraints already covered under equality_constraints.
7. Comments: Include a Python comment before each key-value pair, explaining the rationale behind the constraint.

Return only the Python dictionary update (i.e.,  formalization_dict['inequality_constraints'] = {...}) following these requirements.
Important: Think carefully of inequalities constraint that are not explicit in the problem description that should be considered. If after thinking you conclude the problem contains only equality constraints and no inequality constraints, return: formalization_dict['inequality_constraints'] = {None: None}.
"""

INST_RANK = \
"""
I have a problem in operational research:
  
------ 
###PROBLEM DESCRIPTION###
------

I have the following formalization:

formalization_dict = {
'parameters': {},
'decision_variables': {},
'objective': {},
'equality_constraints': {},
'inequality_constraints': {},
}

You are an expert in optimization modeling. Using the formalization_dict as your current progress, you are tasked with selecting the optimal #VARIABLE# from the provided options.

Please follow these steps:
1. Evaluate the Fit: Determine how well each potential #VARIABLE# reflects the mathematical description in the problem description.
2. Analyze Carefully: Assess each #VARIABLE based on its alignment with the problem description and mathematical formalization.
3. Rank the Options: Rank the variables from best to worst, prioritizing accuracy respect to the problem description.

Present your rankings in the following format:

###
rank = {
1: solution_1,
...,
n: solution_n}
###

Where:
- solution_1 represents the best #VARIABLE#.
- solution_n represents the least suitable #VARIABLE#.

Important: Think carefully STEP BY STEP about your ranking decision. Then conclude by listing the solutions in string format as structured above.

Here are the possible solutions:

solutions = {}
"""








INST_RANK_2 = \
"""
I have a problem in operational research:
  
------ 
###PROBLEM DESCRIPTION###
------

I have the following formalization:

formalization_dict = {
'parameters': {},
'decision_variables': {},
'objective': {},
'equality_constraints': {},
'inequality_constraints': {},
}

You are an expert in optimization modeling. Using the formalization_dict as your current progress, you are tasked with selecting the optimal #VARIABLE# from the provided options.

Please follow these steps:
1. The most important factor if the #VARIABLE# correctly reflect the mathematical description provided in the text in the problem description.
2. Carefully evaluate each potential #VARIABLE#.
3. Rank the variables from best to worst based on their suitability.

Present your rankings in the following format:

###
rank = {
1: solution_1,
...,
n: solution_n}
###

Where:
- solution_1 represents the best #VARIABLE#.
- solution_n represents the least suitable #VARIABLE#.

Important: Think carefully STEP BY STEP about your ranking decision. Then conclude by listing the solutions in string format as structured above.

Here are the possible solutions:

solutions = {}
"""




INST_GROUP_DV = \
"""
I have a problem in operational research:
  
------ 
###PROBLEM DESCRIPTION###
------

I have the following formalization:

formalization_dict = {
'parameters': {},
'decision_variables': {},
'objective': {},
'equality_constraints': {},
'inequality_constraints': {},
}

-Objective:
As an expert in optimization modeling, your role is to evaluate multiple sets of decision variables provided for an operations research problem. You are responsible for determining if two or more sets of decision variables should be grouped together based on their equivalency from an optimization perspective.

-Task Breakdown:
Your grouping decision is critical for assisting a subsequent optimization expert, who will define the objective function, equality constraints, and inequality constraints for each group. To facilitate this process, follow these precise guidelines:

-Equivalency Criteria:
1. Same Objectives and Constraints: Two sets of decision variables should be group together if they result in the definition of the same objective function, equality constraints, and inequality constraints, even if the variable names differ.
2. Conceptual Equivalency: Variable sets should be group together if, despite having different variable names, they define the same underlying concepts that ultimately lead to identical objectives and constraints (both equality and inequality).
3. Non-Equivalency Conditions: Two sets of decision variables should not be grouped toguether if they lead to differences in any of the following: Objective function, Equality constraints, Inequality constraints.
4. Naming Convention Irrelevance: The names of the decision variables are irrelevant for grouping purposes. Only the functional impact of the variables on the objective function and constraints should be considered. If two sets of variables lead to the same results, group them together, even if the names differ.

By following these guidelines, you will help ensure that decision variable sets are clearly classified for the next expert in the process.

Please list your clusters as follows:

###
groups = {
1: group_1,
...,
n: group_n}
###

Where group_i is a python list containing the names (string) of all the set of decision variables that are equivalent. One set of decision variables can only belong to one group. The list should consider at least one element. 
Important: Think carefully STEP BY STEP about your grouping decision, then conclude your assessment using the structured format provided above.

Here are the current solutions:

solutions = {}
"""




INST_RANK_FINAL = \
"""
I have a problem in operational research:
  
------ 
###PROBLEM DESCRIPTION###
------

You are provided with two potential solutions to an operational research problem, both implemented in Python using Gurobipy:

possible_solution_1 = {}


possible_solution_2 = {}

As an expert in optimization modeling, your task is to evaluate and rank these solutions. Please present your rankings in the following format:

###
rank = {
1: S1,
2: S2}
###

Where S1 corresponds to the best solution, and S2 is the second-best solution, both representing the code variable as strings.
To determine the rankings, consider the following criteria:
- How closely each solution aligns with the objectives described in the initial problem.
- How effectively each solution satisfies the requirements outlined in the problem description.

After evaluating, conclude by listing the solutions in string format as structured above.

"""



INST_RANK_FINAL_2 = \
"""
I have a problem in operational research:
  
------ 
###PROBLEM DESCRIPTION###
------

You are provided with two potential solutions to an operational research problem, both implemented in Python using Gurobipy:

possible_solution_1 = {}


possible_solution_2 = {}

As an expert in optimization modeling, your task is to evaluate and rank these solutions. Please present your rankings in the following format:

###
rank = {
1: S1,
2: S2}
###

Where S1 corresponds to the best solution, and S2 is the second-best solution, both representing the code variable as strings.
To determine the rankings, consider the following criteria:
- How closely each solution aligns with the objectives described in the initial problem.
- How effectively each solution satisfies the requirements outlined in the problem description.

After evaluating, conclude by listing the solutions in string format as structured above.

"""



INST_RANK_FINAL_STEP = \
"""
I have a problem in operational research:
  
------ 
###PROBLEM DESCRIPTION###
------

You are provided with two potential solutions to an operational research problem, both implemented in Python using Gurobipy:

possible_solution_A = {}


possible_solution_B = {}

As an expert in optimization modeling, your task is to evaluate and rank these solutions. Please present your rankings in the following format:

###
rank = {
1: S1,
2: S2}
###

Your task, as an expert in optimization modeling, is to evaluate and rank these solutions. Please assess the solutions based on the following criteria:

1- Constraint Satisfaction: Evaluate how effectively each solution satisfies all constraints specified in the problem description.
2- Decision Variable Definition: Assess the appropriateness of the decision variables selected, considering their relevance to modeling the optimization problem.
3- Objective Alignment: Determine how closely each solution aligns with the stated objectives of the problem.
4- Solver Compatibility: Consider how well each solution is suited for its implemenation using optimization solvers like Gurobi, CPLEX, or similar tools.

After evaluating, conclude by listing the solutions in string format as structured above.
When filling S1 or S2 you are only allowed to put the letter 'A' or 'B' referring to the respective solution.

"""

INST_RANK_FINAL_STEP_2 = \
"""
I have a problem in operational research:
  
------ 
###PROBLEM DESCRIPTION###
------

You are provided with two potential solutions to an operational research problem, both to be implemented in Python using Gurobipy:

solution_A = {}

solution_B = {}

As an expert in optimization modeling, your task is to evaluate and determine the best solution from the provided options. Please present your evaluation in the following format:
###
best_solution = {
1: final_solution}
###

Please assess the solutions based on the following criteria:

1- Constraint Satisfaction: Evaluate how effectively each solution satisfies all constraints specified in the problem description.
2- Decision Variable Definition: Assess the appropriateness of the decision variables selected, considering their relevance to modeling the optimization problem.
3- Objective Alignment: Determine how closely each solution aligns with the stated objectives of the problem.

After evaluating, conclude by listing the solutions in string format as structured above.
When filling final_solution you are only allowed to put the letter 'solution_A' or 'solution_B' referring to the respective solution.

"""


INST_SCORE_FINAL = \
"""
I have a problem in operational research:
  
------ 
###PROBLEM DESCRIPTION###
------

You are provided with one potential solution to an operational research problem implemented in Python using Gurobipy:

possible_solution = {}

As an expert in optimization modeling, your task is to evaluate and rank these solutions. Please present your rankings in the following format:

###
score = {
"possible_solution": THIS_SCORE}
###

THIS_SCORE represents a score ranging from 1 to 100, where 100 indicates a perfect solution. To determine the score, evaluate the solution based on the following criteria:

1- Constraint Satisfaction: Evaluate how effectively the solution satisfies all constraints specified in the problem description.
2- Decision Variable Definition: Assess the appropriateness of the decision variables selected, considering their relevance to modeling the optimization problem.
3- Objective Alignment: Determine how closely the solution aligns with the stated objectives of the problem.

Once you have evaluated the solution, conclude your assessment using the structured format provided above.

"""




INST_SCORE_FINAL_PARTIAL = \
"""
I have a problem in operational research:
  
------ 
###PROBLEM DESCRIPTION###
------

I have the following formalization:

formalization_dict = {
'parameters': {},
'decision_variables': {},
'objective': {},
'equality_constraints': {},
'inequality_constraints': {},
}

You are an expert in optimization modeling. Using the formalization_dict as your current progress, you are tasked to estimate an SCORE from by evaluating ONLY the #VARIABLE# from this partial solution.

###
score = {
"possible_solution": THIS_SCORE}
###

THIS_SCORE represents a score ranging from 1 to 100, where 100 indicates a perfect (partial) solution for #VARIABLE#. To determine the score, evaluate the solution based on the following criteria:

The most important factor if the partial solution correctly reflect the mathematical description provided in the text in the problem description until that point.

Once you have evaluated the solution, conclude your assessment using the structured format provided above.

"""