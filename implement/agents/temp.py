from datasets import load_dataset
import random
import json
from constraints_generator import revise_query
import ast

# Load dataset
dataset = load_dataset('osunlp/TravelPlanner', 'validation')['validation']

def generate_new_constraints():
    """Generate a random set of constraints."""
    constraints = []
    
    # Budget constraint
    budget = random.randint(1, 3) * 10
    # budget_type = random.choice(["increase", "decrease"])
    budget_type = "decrease"
    constraints.append({"budget": (budget, budget_type)})
    
    # House rule constraint
    house_rule = random.sample(['pets', 'smoking', 'children under 10', 'visitors', 'parties'], random.randint(1, 3))
    constraints.append({"house rule": (house_rule, "set")})
    
    # Cuisine constraint
    cuisine = random.sample(['Fast Food', 'Italian', 'Bakery', 'Mexican', 'Chinese', 'Japanese', 'French', 'Indian', 'Cafe', 'BBQ','Mediterranean', 'American', 'Desserts', 'Tea', 'Seafood'], random.randint(1, 3))
    constraints.append({"cuisine": (cuisine, "set")})
    
    # Room type constraint
    room_type = random.choice(['shared room', 'private room', 'entire room'])
    constraints.append({"room type": (room_type, "set")})
    
    # Ratings constraint
    ratings = round(random.uniform(3, 4.5), 1)
    constraints.append({"ratings": (ratings, "set")})
    
    # People number constraint
    people_number = random.randint(1, 2)
    constraints.append({"people_number": (people_number, "add")})
    
    return constraints

# Generate new constraints for each query
# and append it to new dictionary with the same key as original dataset
# new_dataset = []
# for data in dataset:
#     new_constraints = generate_new_constraints()
#     data['new_constraints'] = new_constraints
#     # import pdb; pdb.set_trace()
#     new_dataset.append(data)
    
# # Save new dataset
# with open('/home/juhyun/FlexibleReasoningBench/implement/agents/evaluation/database/train_dataset_with_constraints_tuple.json', 'w') as f:
#     json.dump(new_dataset, f, indent=4)

# print("Done! New dataset with constraints is saved.")

# read the dataset from the above file
# with open(dataset_file, "r") as f:
#     dataset = json.load(f)

# def extract_constraints(data):
#     exclude_keys = {'query', 'level', 'annotated_plan', 'reference_information'}
#     return {k: data[k] for k in data.keys() - exclude_keys}

# set seed
random.seed(42)

# one-constraint removal
# new_dataset_with_constraints_removed = []
# for idx, data in enumerate(dataset):
#     if data['level'] != 'easy':
#         extracted_constraints = extract_constraints(data)
#         # randomly select a constraint to remove
#         keys_to_extract = ['people_number', 'budget', 'house rule', 'room type', 'cuisine']
#         constraint_to_remove = random.choice(keys_to_extract) # TODO: people number라는 조건을 처음에 안 주는게 괜찮은가? 안 주면 평가할 때는 그냥 1인용으로 계산?
#         revised_query = revise_query(data['query'], [constraint_to_remove])
#         data['query'] = revised_query
#         # constraints_to_remove를 제외한 나머지 정보를 저장
#         # data['constraints'] = {k: v for k, v in extracted_constraints.items() if k != constraint_to_remove}
#         data[constraint_to_remove] = None
#         import pdb; pdb.set_trace()
#         data['new_constraints'] = {constraint_to_remove: extracted_constraints[constraint_to_remove]}
#         data['idx'] = idx
#         # new_dataset_with_constraints_removed.append(extract_constraints(data))
#         new_dataset_with_constraints_removed.append(data)


def extract_constraints(data):
    """
    Extract all constraints from the data, including those in local_constraint.
    
    Args:
        data (dict): Input data dictionary containing all constraints
        
    Returns:
        dict: Dictionary of all constraints, flattened
    """
    exclude_keys = {'query', 'level', 'annotated_plan', 'reference_information'}
    constraints = {k: data[k] for k in data.keys() - exclude_keys}
    
    # Handle local_constraint if it exists
    if 'local_constraint' in constraints:
        # Convert string representation to dictionary if needed
        if isinstance(constraints['local_constraint'], str):
            local_constraints = ast.literal_eval(constraints['local_constraint'])
        else:
            local_constraints = constraints['local_constraint']
            
        # Add local constraints to main constraints
        constraints.update(local_constraints)
        del constraints['local_constraint']
    
    return constraints

def update_local_constraints(data, constraint_to_remove):
    """
    Update local_constraint dictionary when removing a constraint.
    
    Args:
        data (dict): Input data dictionary
        constraint_to_remove (str): Name of constraint to remove
        
    Returns:
        dict: Updated data dictionary
    """
    local_constraint_keys = ['house rule', 'cuisine', 'room type', 'transportation']
    
    if constraint_to_remove in local_constraint_keys:
        if isinstance(data['local_constraint'], str):
            local_constraints = ast.literal_eval(data['local_constraint'])
        else:
            local_constraints = data['local_constraint']
            
        local_constraints[constraint_to_remove] = None
        data['local_constraint'] = str(local_constraints)
    else:
        data[constraint_to_remove] = None
        
    return data

# Modified dataset processing
new_dataset_with_constraints_removed = []
for idx, data in enumerate(dataset):
    if data['level'] != 'easy':
        # Extract all constraints including those in local_constraint
        extracted_constraints = extract_constraints(data)
        
        # Define possible constraints to remove
        keys_to_extract = ['people_number', 'budget', 'house rule', 'room type', 'cuisine']
        global_constraints = ['people_number', 'budget']
        local_constraints = ['house rule', 'room type', 'cuisine']
        
        while True:
            constraint_to_remove = random.choice(keys_to_extract)
            
            # Check if the constraint to remove is in local_constraint and not None
            if constraint_to_remove in ['house rule', 'room type', 'cuisine']:
                if isinstance(data['local_constraint'], str):
                    local_constraints = ast.literal_eval(data['local_constraint'])
                else:
                    local_constraints = data['local_constraint']
                if local_constraints.get(constraint_to_remove) is not None:
                    break
            else:
                if data.get(constraint_to_remove) is not None:
                    break
        
        # Store original value of the removed constraint
        if constraint_to_remove in ['house rule', 'room type', 'cuisine']:
            original_value = local_constraints.get(constraint_to_remove)
        else:
            original_value = data.get(constraint_to_remove)
            
        # Update constraints in data
        data = update_local_constraints(data, constraint_to_remove)
        
        # Update query
        revised_query = revise_query(data['query'], [constraint_to_remove+':'+str(original_value)])
        data['query'] = revised_query

        # Store the removed constraint and its original value
        data['new_constraints'] = [{constraint_to_remove: original_value}]
        data['idx'] = idx
        
        new_dataset_with_constraints_removed.append(data)

# Save new dataset
with open('/home/juhyun/FlexibleReasoningBench/implement/agents/evaluation/database/validation_dataset_with_one_constraint_removed.json', 'w') as f:
    json.dump(new_dataset_with_constraints_removed, f, indent=4)

print(f"Done! New dataset with one constraint removed is saved.")


