import json
from collections import defaultdict

def track_constraint_changes(results, change_const=None):
    # Add counters for debugging
    skipped_single_turn = 0
    skipped_none_values = 0
    total_comparisons = 0
    
    change_counters = {
        'budget': defaultdict(int),
        'cuisine': defaultdict(int),
        'room_type': defaultdict(int),
        'room_rule': defaultdict(int)
    }
    llama_fp = open(f"/mnt/nas2/juhyun/FlexibleReasoningBench/implement/results/two_turn/results_{change_const}_20250129_llama_reevaluated_for_track_change.json", "r")
    llama_results = json.load(llama_fp)
    llama_fp.close()
    valid_idxs = set()
    for idx, result in llama_results.items():
        valid_idxs.add(idx)

    for idx, result in results.items():
        if idx not in valid_idxs:
            continue
        turns = result["detailed_results"]
        
        if len(turns) < 2:
            skipped_single_turn += 1
            continue
            
        for i in range(len(turns) - 1):
            turn1 = turns[i]
            turn2 = turns[i + 1]
            
            for constraint_type in ['budget', 'cuisine', 'room_type', 'room_rule']:
                prev_result = turn1['constraint_scores'][constraint_type]['is_correct']
                curr_result = turn2['constraint_scores'][constraint_type]['is_correct']
                
                if prev_result is None or curr_result is None:
                    skipped_none_values += 1
                    continue
                
                total_comparisons += 1
                
                if prev_result == True and curr_result == False:
                    change_counters[constraint_type]['True2False'] += 1
                elif prev_result == False and curr_result == True:
                    change_counters[constraint_type]['False2True'] += 1
                elif prev_result == True and curr_result == True:
                    change_counters[constraint_type]['AlreadyTrue'] += 1
                elif prev_result == False and curr_result == False:
                    change_counters[constraint_type]['AlwaysFalse'] += 1
    
    # print(f"Comparisons skipped due to None values: {skipped_none_values}")
    # print(f"Total valid comparisons made: {total_comparisons}")
            
    return change_counters

date = "20250126"
for changed_constraint in ['cuisine', 'room_type', 'house_rule']:
    print(f"Changed Constraint: {changed_constraint}")
    result_file_path = f"/mnt/nas2/juhyun/FlexibleReasoningBench/implement/results/two_turn/results_{changed_constraint}_20250126_gpt_reevaluated_for_track_change.json"
    # ref_file_path = "/mnt/nas2/juhyun/FlexibleReasoningBench/implement/agents/evaluation/database/validation_ref_info.jsonl"
    with open(result_file_path, "r") as f:
        results = json.load(f)
        
    change_stats = track_constraint_changes(results, change_const=changed_constraint)

    # Print results
    for constraint_type, stats in change_stats.items():
        print(f"Constraint Type: {constraint_type}")
        # I want the change_type order to be the same every time
        for change_type in ['True2False', 'False2True', 'AlreadyTrue', 'AlwaysFalse']:
            print(f"{change_type}: {stats[change_type]}")
        print("\n")

def track_constraint_changes_preference(results):
    # Add counters for debugging
    skipped_single_turn = 0
    skipped_none_values = 0
    total_comparisons = 0
    
    # change_counters = {
    #     'budget': defaultdict(int),
    #     'cuisine': defaultdict(int),
    #     'room_type': defaultdict(int),
    #     'room_rule': defaultdict(int)
    # }
    change_counters = {
        'budget': defaultdict(int)
    }
    
    for idx, result in results.items():
        turns = result["detailed_results"]
        
        if len(turns) < 2:
            skipped_single_turn += 1
            continue
            
        for i in range(len(turns) - 1):
            turn1 = turns[i]
            turn2 = turns[i + 1]
            
            for constraint_type in ['budget', 'cuisine', 'room_type', 'room_rule']:
            # for constraint_type in ['budget']:
                prev_result = turn1['constraint_scores'][constraint_type]['is_correct']
                curr_result = turn2['constraint_scores'][constraint_type]['is_correct']
                
                if prev_result is None or curr_result is None:
                    skipped_none_values += 1
                    continue
                
                total_comparisons += 1
                
                if prev_result == True and curr_result == False:
                    change_counters[constraint_type]['True2False'] += 1
                elif prev_result == False and curr_result == True:
                    change_counters[constraint_type]['False2True'] += 1
                elif prev_result == True and curr_result == True:
                    change_counters[constraint_type]['AlreadyTrue'] += 1
                elif prev_result == False and curr_result == False:
                    change_counters[constraint_type]['AlwaysFalse'] += 1
    
    # print(f"Comparisons skipped due to None values: {skipped_none_values}")
    # print(f"Total valid comparisons made: {total_comparisons}")
            
    return change_counters

# for budget_size in ['small', 'middle', 'high']:
#     full_stats = {
#         'True2False': 0,
#         'False2True': 0,
#         'AlreadyTrue': 0,
#         'AlwaysFalse': 0
#     }
    
#     preference_results = {}
#     for preference_type in ['rating', 'cuisine']:
#         # print(f"Budget Size: {budget_size}, Preference: {preference_type}")
#         file_path = f"/mnt/nas2/juhyun/FlexibleReasoningBench/implement/results/preference/results_{budget_size}_{preference_type}_20250130_llama_preference_reevaluated_for_track_change.json"
#         with open(file_path, "r") as f:
#             results = json.load(f)
        
#         change_stats = track_constraint_changes_preference(results)
#         preference_results[preference_type] = change_stats

#         # Print individual results
#         # for constraint_type, stats in change_stats.items():
#         #     print(f"Constraint Type: {constraint_type}")
#         #     for change_type in ['True2False', 'False2True', 'AlreadyTrue', 'AlwaysFalse']:
#         #         print(f"{change_type}: {stats[change_type]}")
#         #     print("\n")

#     # Aggregate results from both preference types
#     for change_type in ['True2False', 'False2True', 'AlreadyTrue', 'AlwaysFalse']:
#         for preference_type in ['rating', 'cuisine']:
#             full_stats[change_type] += preference_results[preference_type]['budget'][change_type]

#     print("Full Stats:")
#     for change_type in ['True2False', 'False2True', 'AlreadyTrue', 'AlwaysFalse']:
#         print(f"{change_type}: {full_stats[change_type]}")
#     print("\n")
    


def track_constraint_changes_three_turn(results):
    # Add counters for debugging
    skipped_single_turn = 0
    skipped_none_values = 0
    total_comparisons_1_to_2 = 0
    total_comparisons_2_to_3 = 0

    # Separate change counters for 1->2 and 2->3
    change_counters_1_to_2 = {
        'budget': defaultdict(int),
        'cuisine': defaultdict(int),
        'room_type': defaultdict(int),
        'room_rule': defaultdict(int)
    }
    change_counters_2_to_3 = {
        'budget': defaultdict(int),
        'cuisine': defaultdict(int),
        'room_type': defaultdict(int),
        'room_rule': defaultdict(int)
    }

    for idx, results_list in results.items():
        # print(f"Processing idx: {idx}")

        # Process each result in the list of results for the current idx
        for const_type_idx, result in enumerate(results_list):
            # print(f"Processing constraint type index: {const_type_idx}")
            turns = result["detailed_results"]

            if len(turns) < 3:
                skipped_single_turn += 1
                continue

            # Compare 1->2
            for i in range(2):  # First for turn 1->2 (i=0), then for turn 2->3 (i=1)
                turn1 = turns[i]
                turn2 = turns[i + 1]
                change_counters = change_counters_1_to_2 if i == 0 else change_counters_2_to_3

                for constraint_type in ['budget', 'cuisine', 'room_type', 'room_rule']:
                    prev_result = turn1['constraint_scores'][constraint_type]['is_correct']
                    curr_result = turn2['constraint_scores'][constraint_type]['is_correct']

                    if prev_result is None or curr_result is None:
                        skipped_none_values += 1
                        continue

                    # Track total comparisons for debugging
                    if i == 0:
                        total_comparisons_1_to_2 += 1
                    else:
                        total_comparisons_2_to_3 += 1

                    if prev_result == True and curr_result == False:
                        change_counters[constraint_type]['True2False'] += 1
                    elif prev_result == False and curr_result == True:
                        change_counters[constraint_type]['False2True'] += 1
                    elif prev_result == True and curr_result == True:
                        change_counters[constraint_type]['AlreadyTrue'] += 1
                    elif prev_result == False and curr_result == False:
                        change_counters[constraint_type]['AlwaysFalse'] += 1

    return change_counters_1_to_2, change_counters_2_to_3

# date = "20250130"
# for changed_constraint in ['global_local', 'local_global']:
#     print(f"Changed Constraint: {changed_constraint}")
#     result_file_path = f"/mnt/nas2/juhyun/FlexibleReasoningBench/implement/results/three_turn/results_{changed_constraint}_{date}_gpt_reevaluated_for_track_change.json"
#     with open(result_file_path, "r") as f:
#         results = json.load(f)

#     change_stats_1_to_2, change_stats_2_to_3 = track_constraint_changes_three_turn(results)

#     # Print results for 1->2
#     print("Turn 1 -> Turn 2 Changes:")
#     for constraint_type, stats in change_stats_1_to_2.items():
#         print(f"Constraint Type: {constraint_type}")
#         for change_type in ['True2False', 'False2True', 'AlreadyTrue', 'AlwaysFalse']:
#             print(f"{change_type}: {stats[change_type]}")
#         print("\n")

#     # Print results for 2->3
#     print("Turn 2 -> Turn 3 Changes:")
#     for constraint_type, stats in change_stats_2_to_3.items():
#         print(f"Constraint Type: {constraint_type}")
#         for change_type in ['True2False', 'False2True', 'AlreadyTrue', 'AlwaysFalse']:
#             print(f"{change_type}: {stats[change_type]}")
#         print("\n")

def track_constraint_changes_preference(results):
    # Add counters for debugging
    skipped_single_turn = 0
    skipped_none_values = 0
    total_comparisons_1_to_2 = 0
    total_comparisons_2_to_3 = 0

    # Separate change counters for 1->2 and 2->3
    change_counters_1_to_2 = {
        'budget': defaultdict(int),
        'cuisine': defaultdict(int),
        'room_type': defaultdict(int),
        'room_rule': defaultdict(int)
    }

    for idx, results_list in results.items():
        # print(f"Processing idx: {idx}")

        # Process each result in the list of results for the current idx
        for const_type_idx, result in enumerate(results_list):
            # print(f"Processing constraint type index: {const_type_idx}")
            import pdb; pdb.set_trace()
            turns = result["detailed_results"]

            if len(turns) < 2:
                skipped_single_turn += 1
                continue

            # Compare 1->2
            for i in range(2):  # First for turn 1->2 (i=0), then for turn 2->3 (i=1)
                turn1 = turns[i]
                turn2 = turns[i + 1]
                change_counters = change_counters_1_to_2 

                for constraint_type in ['budget', 'cuisine', 'room_type', 'room_rule']:
                    prev_result = turn1['constraint_scores'][constraint_type]['is_correct']
                    curr_result = turn2['constraint_scores'][constraint_type]['is_correct']

                    if prev_result is None or curr_result is None:
                        skipped_none_values += 1
                        continue

                    # Track total comparisons for debugging
                    if i == 0:
                        total_comparisons_1_to_2 += 1
                    else:
                        total_comparisons_2_to_3 += 1

                    if prev_result == True and curr_result == False:
                        change_counters[constraint_type]['True2False'] += 1
                    elif prev_result == False and curr_result == True:
                        change_counters[constraint_type]['False2True'] += 1
                    elif prev_result == True and curr_result == True:
                        change_counters[constraint_type]['AlreadyTrue'] += 1
                    elif prev_result == False and curr_result == False:
                        change_counters[constraint_type]['AlwaysFalse'] += 1

    return change_counters_1_to_2

