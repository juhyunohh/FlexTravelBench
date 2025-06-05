from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Union
from dataclasses import dataclass
# import random
from utils import inference_gpt, MODEL_PATHS
# set the path to "FlexibleReasoningBench/implement/agents" directory
import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class TravelConstraint:
    category: str  # 'budget', 'house_rule', 'cuisine', etc.
    value: Any
    description: str


# Given query, and a constraint, revise the query by removing the constraint, 
# and return 1. revised query without the constraint 2. another query that only contains the constraint

def revise_query(query: Dict, constraint: List) -> str:
    """
    Revise the query by removing the constraint.

    Args:
        query (Dict): The initial query and constraint
        constraint (List): The constraint to remove, e.g., {'budget': (20, 'decrease')}.

    Returns:
        revised_query_without_constraint (str): The revised query without the constraint.
        updated_constraints (Dict): The updated constraints with the query removed.

    """
    prompt = f"Given the following query and constraint, revise the original query so that the given constraints are removed:\n\nQuery:\n{query}\n\nConstraint:\n{constraint}.\n\nRevised Query:"
    revised_query = inference_gpt("gpt-4o", prompt)
    revised_query = revised_query.replace("Revised Query:", "").strip()
    return revised_query
    
# Example usage
# query = "Please help me plan a trip from St. Petersburg to Rockford spanning 3 days from March 16th to March 18th, 2022. The travel should be planned for a single person with a budget of $1,700."
# constraint = ['budget']
# prompt = f"Given the following query and constraint, revise the original query so that the constraint is removed:\n\nQuery:\n{query}\n\nConstraint:\n{str(constraint)}."
# revised_query_without_constraint = revise_query(query, constraint)

# function to generate a query based on the constraints, with the value already set
def generate_query(new_constraints) -> str:
    """
    Generate a query based on the given new constratint. Just a single sentence that would a follow-up query of the user.

    Args:
        constraints (Dict): The constraints to generate a query for.
    Returns:
        query (str): The generated query.
    """
    query = ""
    for category, value in new_constraints.items():
        if category == "budget":
            query += f"My budget is {value}."
        elif category == "house rule":
            query += f"The accommodation should allow for {value}."
        elif category == "cuisine":
            value = ", ".join(value) if isinstance(value, (list, tuple)) else value
            query += f"we're particularly interested in trying out {value} food."
        elif category == "room type":
            query += f"I need a {value}."
        elif category == "ratings":
            query += f"All restaurants must have a minimum rating of {value}."
        elif category == "people_number":
            # query += f"I will be traveling with {value-1} people."
            query += f"The total number of people traveling is {value}."
        elif category == "cuisine_pref":
            query += f"I prefer to try a {value[0]} place at least {value[1]} times if possible."
        elif category == "rating_pref":
            query += f"I prefer to visit restaurants with a minimum rating of {value[0]} at least {value[1]} times if possible."
    return query

def update_constraints_and_query(
    base_query_constraint: Dict,
    constraint: Dict[str, Union[Tuple[Any, str], Any]]
) -> Tuple[Dict, str]:
    """
    Update constraints and query based on the input constraint.
    Handles both tuple-style constraints (value, action) and direct value assignments.

    Args:
        base_query_constraint (Dict): The initial query and constraints.
        constraint (Dict): The constraint to modify. Can be either:
            - Tuple format: {'budget': (20, 'decrease')}
            - Direct value format: {'budget': 5000}

    Returns:
        Tuple[Dict, str]: Updated constraint set and query.
    """
    updated_query_constraint = base_query_constraint.copy()
    local_constraints = updated_query_constraint.get("local_constraint", {})
    if type(local_constraints) is not dict and local_constraints is not None:
        local_constraints = eval(local_constraints)

    # Handle direct value assignments using generate_query
    def contains_keywords(value, keywords):
        if isinstance(value, list):
            return any(k in value for k in keywords)
        return False

    keywords_to_check = {"set", "add", "decrease", "increase"}

    if not any(contains_keywords(v, keywords_to_check) for v in constraint.values()):
        query_description = generate_query(constraint)

    # if all(not isinstance(v, list) for v in constraint.values()):
    #     query_description = generate_query(constraint)
        
        # Update constraints
        for key, value in constraint.items():
            if key in ["house_rule", "house rule", "room rule"]:
                local_constraints["house rule"] = value
            elif key in ["room_type", "room type"]:
                local_constraints["room type"] = value
            elif key in ["cuisine", "transportation"]:
                local_constraints[key] = value
            else:
                updated_query_constraint[key] = value
        
        updated_query_constraint["local_constraint"] = local_constraints
        return updated_query_constraint, query_description

    # Original tuple-based constraint handling
    query_description = ""
    for key, value_action in constraint.items():
        # if not isinstance(value_action, tuple):
        #     raise ValueError(f"Mixed constraint formats not supported. All constraints must be either tuples or direct values.")
        
        value, action = value_action
        if key == "budget":
            current_budget = updated_query_constraint.get("budget", 0)
            if action == "increase":
                new_budget = int(current_budget * (1 + value / 100))
            elif action == "decrease":
                new_budget = int(current_budget * (1 - value / 100))
            else:
                raise ValueError("Invalid action for budget constraint")
            updated_query_constraint["budget"] = new_budget
            query_description = f"My budget has changed to {new_budget}."

        elif key == "people_number":
            current_size = updated_query_constraint.get("people_number", 1)
            if action == "add":
                new_size = current_size + value
            elif action == "subtract":
                new_size = max(1, current_size - value)
            else:
                raise ValueError("Invalid action for people_number constraint")
            updated_query_constraint["people_number"] = new_size
            query_description = f"The number of people traveling has changed to {new_size}."

        elif key in ["house_rule", "room rule", "house rule"]:
            if action == "set":
                local_constraints["house rule"] = value
            else:
                raise ValueError("Invalid action for house_rule constraint")
            query_description = f"Accommodation should allow for {value}."

        elif key == "cuisine":
            if action == "set":
                local_constraints["cuisine"] = value
            else:
                raise ValueError("Invalid action for cuisine constraint")
            cuisines_str = ", ".join(value) if isinstance(value, (list, tuple)) else value
            query_description = f"Preferred cuisines: {cuisines_str}."

        elif key == "transportation":
            if action == "set":
                local_constraints["transportation"] = value
            else:
                raise ValueError("Invalid action for transportation constraint")
            query_description = f"Transportation requirement: {value}."

        elif key in ["room_type", "room type"]:
            if action == "set":
                local_constraints["room type"] = value
            else:
                raise ValueError("Invalid action for room_type constraint")
            query_description = f"Required room type: {value}."

        elif key == "ratings":
            if action == "set":
                updated_query_constraint["ratings"] = value
            else:
                raise ValueError("Invalid action for ratings constraint")
            query_description = f"All restaurants must have a minimum rating of {value:.1f}."

        else:
            raise ValueError(f"Unknown constraint category: {key}")

    updated_query_constraint["local_constraint"] = local_constraints
    return updated_query_constraint, query_description


def find_min_restaurant(data, cuisine=None, rating=None):

    df = pd.DataFrame(data[0])
    df.columns = df.columns.str.lower()
    df['cuisines'] = df['cuisines'].apply(lambda x: x if isinstance(x, list) else x.split(', '))

    if cuisine:
        df = df[df['cuisines'].apply(lambda x: any(c in x for c in (cuisine if isinstance(cuisine, list) else [cuisine])))]
    
    if rating is not None:
        df = df[df['aggregate rating'] >= rating]
    
    if not df.empty:
        min_cost_row = df.loc[df['average cost'].idxmin()]
        return min_cost_row[['name', 'average cost']].to_dict()
    else:
        return "No restaurant found with given criteria."



def find_min_room(data, room_type=None, house_rule=None, city=None):
    df = pd.DataFrame(data[0])
    df.columns = df.columns.str.lower()
    room_type_filters = {
        'not shared room': lambda x: x.lower() != 'shared room',
        'shared room': lambda x: x.lower() == 'shared room',
        'private room': lambda x: x.lower() == 'private room',
        'entire room': lambda x: x.lower() == 'entire home/apt'
    }
    rule_map = {
        'smoking': 'No smoking',
        'parties': 'No parties',
        'children under 10': 'No children under 10',
        'visitors': 'No visitors',
        'pets': 'No pets'
    }
    
    if room_type and (room_type.lower() in room_type_filters):
        df = df[df['room type'].apply(room_type_filters[room_type.lower()])]

    if house_rule and house_rule:
      #  df = df[df['house_rules'].str.contains(house_rule, case=False, na=False)]
        exclude_condition = rule_map[house_rule].lower() 
        df = df[~df['house_rules'].str.lower().eq(exclude_condition)]  

    if city:
        df = df[df['city'].str.contains(city, case=False, na=False)]
    
    if not df.empty:
        min_price_row = df.loc[df['price'].idxmin()]
        return min_price_row[['name', 'price', 'city']].to_dict()
    else:
        print("No listing found with given criteria.")
        return {'name':None,'price':None,'city':None}

def generate_preference_query(constraints: dict,reference):
    
    local_constraint = eval(constraints['local_constraint'])
    local_constraint['days'] = constraints['days']
    local_constraint['budget'] = constraints['budget']
    local_constraint['people_number'] = constraints['people_number']
    new_constraint = constraints['new_constraints'][0]
    def merge_constraints(local_constraint, new_constraint):
        merged = {}
        all_keys = set(local_constraint) | set(new_constraint)
        
        for key in all_keys:
            val1 = local_constraint.get(key)
            val2 = new_constraint.get(key)
            
            if val1 is not None:
                merged[key] = val1
            elif val2 is not None:
                merged[key] = val2
        
        return merged
    constraint = merge_constraints(local_constraint, new_constraint)
    room_type = constraint.get("room type", None)
    budget = constraint.get("budget", None)
    city = constraint.get("city", None)
    days = constraint.get("days", None)
    people_number = constraint.get("people_number", None)
    cuisine = constraint.get("cuisine", None)
    house_rule = constraint.get("house_rule", None)
    min_room = find_min_room(reference['room'], room_type=room_type, house_rule=None, city=None)
    min_resturant = find_min_restaurant(reference['cuisine'], cuisine=None, rating=None)
    try:
        min_room_budget = min_room['price'] * people_number * days
    except:
        import pdb;pdb.set_trace()
    budget_remain = budget - min_room_budget
    #generate cuisine preference



    cuisine_min = find_min_restaurant(reference['cuisine'])
    cuisine_max = find_max_cuisine(reference['cuisine'],cuisine)
    cuisine_max_name = cuisine_max['name']   
    #find N
    N_meal = (days-2) *3 + 2
    
    if not cuisine:

        N_cuisine = find_min_N_cuisine_max(budget_remain, N_meal, cuisine_max['average cost'], cuisine_min['average cost'])
        if N_cuisine != None:
            cuisine_output = (cuisine_max_name,N_cuisine)
        else:
            cuisine_output = None
    else:
        cuisine_output = None

    #generate rating preference  
    N_rating,rating = find_min_N_rating_max(budget_remain, N_meal, cuisine_min['average cost'], reference['cuisine'])
    if N_rating:
        rating_output = (N_rating,rating)
    else:
        rating_output = None
    return {'cuisine':cuisine_output,'rating':rating_output}   
    


def find_max_cuisine(data, cuisines=None):
    df = pd.DataFrame(data[0])
    df.columns = df.columns.str.lower()
    # Convert cuisines column from string to list
    df['cuisines'] = df['cuisines'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
    
    cuisine_prices = {}
    for _, row in df.iterrows():
        for cuisine in row['cuisines']:
            if cuisines is None or cuisine in cuisines:
                if cuisine not in cuisine_prices or row['average cost'] < cuisine_prices[cuisine]:
                    cuisine_prices[cuisine] = row['average cost']
    
    if not cuisine_prices:
        return None
    
    # Find the cuisine with the highest minimum price
    max_cuisine = max(cuisine_prices, key=cuisine_prices.get)
    return {'name':max_cuisine,'average cost' :cuisine_prices[max_cuisine]}


def find_min_N_cuisine_max(budget, N_meal, cuisine_max, cuisine_min):

    for N_cuisine_max in range(2,N_meal + 1):
        N_cuisine_min = N_meal - N_cuisine_max
        total_cost = N_cuisine_max * cuisine_max + N_cuisine_min * cuisine_min
        
        if total_cost > budget:
            return N_cuisine_max
    
    return None 

def find_min_N_rating_max(budget, N_meal, cuisine_min, reference):
    df = pd.DataFrame(reference[0])
    df.columns = df.columns.str.lower()
    ratings = sorted(df['aggregate rating'].unique())[1:]

    for n in range(2,N_meal + 1):
        for rating in ratings:
            cuisine_max = find_min_restaurant(reference, rating=rating)
            
            # Ensure valid data
            if not isinstance(cuisine_max, dict) or 'average cost' not in cuisine_max:
                continue
            
            cuisine_max_price = cuisine_max['average cost']  # Extracting price
            
            # Ensure cuisine_max_price is a valid number
            if not isinstance(cuisine_max_price, (int, float)):
                continue
            
            total_cost = n * cuisine_max_price + (N_meal - n) * cuisine_min
            
            if total_cost > budget:
                return n,rating  # Return minimum N_cuisine_max that satisfies condition
    
    return None,None  # No valid combination found


