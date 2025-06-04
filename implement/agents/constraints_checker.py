import os
import sys
import json 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.hard_constraint import evaluation as hard_eval


def planning_validate_constraints(turn_constraint, updated_constraints, response_data, ref_data):
    """
    Validates travel constraints based on the current turn's constraint and all updated constraints.
    
    Args:
        turn_constraint (str): Current turn's constraint (e.g., 'budget')
        updated_constraints (dict): All constraints including updates
        response_data (list): List of daily itinerary data
        ref_data (dict): Reference data for the current turn
    
    Returns:
        tuple: (turn_validation_result, all_constraints_result)
            - turn_validation_result: Dictionary containing validation result for the current turn's constraint
            - all_constraints_result: Dictionary containing validation results for all constraints
    """
    # Run evaluation on the entire plan
    all_constraints_result = hard_eval(updated_constraints, response_data, ref_data)

    # Extract the result specific to the current turn's constraint
    if type(turn_constraint) is dict:
       turn_constraint = next(iter(turn_constraint))
    # if turn_constraint == 'people_number':
    #     turn_validation_result = all_constraints_result.get('budget')
    print ("Current Turn Constraint:", turn_constraint)
    if turn_constraint == 'house rule':
        turn_constraint = 'room_rule'
    turn_validation_result = all_constraints_result.get(turn_constraint.replace(" ", "_"), 'No specific constraint this turn')

    return turn_validation_result, all_constraints_result
    