from datasets import load_dataset
import random
import ast
import json
from copy import deepcopy
from constraints_generator import revise_query
from tqdm import tqdm
import os

def extract_constraints(data):
    """
    Extract all constraints from the data, including those in local_constraint.
    """
    exclude_keys = {'query', 'level', 'annotated_plan', 'reference_information'}
    constraints = {k: data[k] for k in data.keys() - exclude_keys}
    
    if 'local_constraint' in constraints:
        if isinstance(constraints['local_constraint'], str):
            local_constraints = ast.literal_eval(constraints['local_constraint'])
        else:
            local_constraints = constraints['local_constraint']
            
        constraints.update(local_constraints)
        del constraints['local_constraint']
    
    return constraints

def update_local_constraints(data, constraint_to_remove):
    """
    Update local_constraint dictionary when removing a constraint.
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

def process_single_constraints(dataset, constraints_list):
    """Process dataset for each single constraint."""
    results = {}
    constraint_counts = {key: 0 for key in constraints_list}
    
    for constraint in tqdm(constraints_list, desc="Processing single constraints"):
        processed_data = []
        
        for idx, data in enumerate(dataset):
            if data['level'] != 'easy':
                current_data = deepcopy(data)
                
                if constraint in ['house rule', 'room type', 'cuisine']:
                    if isinstance(current_data['local_constraint'], str):
                        local_constraints = ast.literal_eval(current_data['local_constraint'])
                    else:
                        local_constraints = current_data['local_constraint']
                        
                    if local_constraints.get(constraint) is None:
                        continue
                        
                    original_value = local_constraints[constraint]
                else:
                    if current_data.get(constraint) is None:
                        continue
                        
                    original_value = current_data[constraint]
                
                current_data = update_local_constraints(current_data, constraint)
                revised_query = revise_query(current_data['query'], 
                                          [constraint+':'+str(original_value)])
                current_data['query'] = revised_query
                current_data['new_constraints'] = [{constraint: original_value}]
                current_data['idx'] = idx
                
                processed_data.append(current_data)
                constraint_counts[constraint] += 1
        
        results[f'dataset_without_{constraint.replace(" ", "_")}_one'] = processed_data
    
    print("\nSingle constraint processing counts:")
    for constraint, count in constraint_counts.items():
        print(f"{constraint}: {count} instances")
    
    return results

def process_constraint_pairs(dataset, global_constraints, local_constraints):
    """Process dataset for constraint pairs."""
    results = {}
    total_combinations = len(global_constraints) * len(local_constraints)
    
    # Process global-first, then local
    global_local_data = []
    with tqdm(total=total_combinations, desc="Processing global-first pairs") as pbar:
        for global_constraint in global_constraints:
            for local_constraint in local_constraints:
                processed_data = []
                
                for idx, data in enumerate(dataset):
                    if data['level'] != 'easy':
                        current_data = deepcopy(data)
                        
                        if local_constraint in ['house rule', 'room type', 'cuisine']:
                            local_const_dict = ast.literal_eval(current_data['local_constraint']) if isinstance(current_data['local_constraint'], str) else current_data['local_constraint']
                            if local_const_dict.get(local_constraint) is None or current_data.get(global_constraint) is None:
                                continue
                            local_value = local_const_dict[local_constraint]
                        else:
                            continue
                            
                        global_value = current_data[global_constraint]
                        
                        current_data = update_local_constraints(current_data, local_constraint)
                        current_data = update_local_constraints(current_data, global_constraint)
                        
                        revised_query = revise_query(current_data['query'], 
                                                  [f"{global_constraint}:{str(global_value)}", 
                                                   f"{local_constraint}:{str(local_value)}"])
                        current_data['query'] = revised_query
                        current_data['new_constraints'] = [
                            {global_constraint: global_value},
                            {local_constraint: local_value}
                        ]
                        current_data['idx'] = idx
                        
                        processed_data.append(current_data)
                
                global_local_data.extend(processed_data)
                pbar.update(1)
    
    results['global_local_dataset'] = global_local_data
    
    # Process local-first, then global
    local_global_data = []
    with tqdm(total=total_combinations, desc="Processing local-first pairs") as pbar:
        for local_constraint in local_constraints:
            for global_constraint in global_constraints:
                processed_data = []
                
                for idx, data in enumerate(dataset):
                    if data['level'] != 'easy':
                        current_data = deepcopy(data)
                        
                        if local_constraint in ['house rule', 'room type', 'cuisine']:
                            local_const_dict = ast.literal_eval(current_data['local_constraint']) if isinstance(current_data['local_constraint'], str) else current_data['local_constraint']
                            if local_const_dict.get(local_constraint) is None or current_data.get(global_constraint) is None:
                                continue
                            local_value = local_const_dict[local_constraint]
                        else:
                            continue
                            
                        global_value = current_data[global_constraint]
                        
                        current_data = update_local_constraints(current_data, local_constraint)
                        current_data = update_local_constraints(current_data, global_constraint)
                        
                        revised_query = revise_query(current_data['query'], 
                                                  [f"{local_constraint}:{str(local_value)}", 
                                                   f"{global_constraint}:{str(global_value)}"])
                        current_data['query'] = revised_query
                        current_data['new_constraints'] = [
                            {local_constraint: local_value},
                            {global_constraint: global_value}
                        ]
                        current_data['idx'] = idx
                        
                        processed_data.append(current_data)
                
                local_global_data.extend(processed_data)
                pbar.update(1)
    
    results['local_global_dataset'] = local_global_data
    
    return results

def generate_dataset(save_path, single_constraints=None, global_constraints=None, local_constraints=None):
    """
    Main function to generate datasets with single constraints and/or constraint pairs.
    
    Args:
        save_path (str): Directory path to save the generated datasets
        single_constraints (list, optional): List of single constraints to process
        global_constraints (list, optional): List of global constraints for pair processing
        local_constraints (list, optional): List of local constraints for pair processing
    """
    # Load the dataset
    dataset = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
    dataset = dataset.select(range(90)) # For testing, limit to first 90 entries
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Process single constraints if provided
    if single_constraints:
        single_results = process_single_constraints(dataset, single_constraints)
        for key, data in single_results.items():
            output_path = os.path.join(save_path, f"{key}.json")
            with open(output_path, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Saved single constraint dataset to: {output_path}")
    
    # Process constraint pairs if both global and local constraints are provided
    if global_constraints and local_constraints:
        pair_results = process_constraint_pairs(dataset, global_constraints, local_constraints)
        for key, data in pair_results.items():
            output_path = os.path.join(save_path, f"{key}.json")
            with open(output_path, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Saved constraint pair dataset to: {output_path}")

if __name__ == "__main__":
    # Example usage
    save_path = "./evaluation/database"
    single_constraints = ['budget', 'house rule', 'room type', 'cuisine']
    # global_constraints = ['budget', 'people_number']
    global_constraints = ['budget']
    local_constraints = ['house rule', 'room type', 'cuisine']
    
    generate_dataset(
        save_path=save_path,
        single_constraints=single_constraints,
        global_constraints=global_constraints,
        local_constraints=local_constraints
    )