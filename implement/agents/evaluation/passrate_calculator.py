import json
from typing import Dict, List, Optional, Set
from collections import defaultdict

class ConstraintEvaluator:
    def __init__(self):
        self.local_constraints = ['house_rule', 'room_type', 'cuisine']
        self.global_constraints = ['budget']
        self.preference_constraints = ['rating_pref', 'cuisine_pref']
        
    def load_json_file(self, file_path: str) -> Dict:
        with open(file_path, "r") as f:
            return json.load(f)
            
    def load_aggregated_results(self, file_path: str) -> Dict:
        aggregated_results = {}
        all_results = self.load_json_file(file_path)
        for k, v in all_results.items():
            if k not in aggregated_results:
                aggregated_results[k] = []
            aggregated_results[k].append(v)
        return aggregated_results

    def calculate_average_pass_rates(self, pass_rates: Dict[str, float]) -> float:
        return round(sum(pass_rates.values()) / len(pass_rates), 2)

class TwoTurnEvaluator(ConstraintEvaluator):
    def __init__(self, date: str, model: str):
        super().__init__()
        self.date = date
        self.model = model
        
    def load_two_turn_results(self) -> Dict:
        aggregated_results = {}
        for constraint in self.local_constraints + self.global_constraints:
            file_path = f"/mnt/nas2/juhyun/FlexibleReasoningBench/implement/results/two_turn/results_{constraint}_{self.date}_{self.model}.json"
            results = self.load_json_file(file_path)
            for k, v in results.items():
                if v['constraint_type'] == constraint:
                    if k not in aggregated_results:
                        aggregated_results[k] = []
                    aggregated_results[k].append(v)
        return aggregated_results

    def calculate_constraint_pass_rates(self, results: Dict) -> Dict:
        pass_rates = {}
        for idx, result in results.items():
            pass_rates[idx] = {
                'house_rule': None,
                'room_type': None,
                'cuisine': None,
                'budget': None,
            }
            for item in result:
                constraint_type = item['constraint_type']
                pass_rates[idx][constraint_type] = item['detailed_results'][-1]['constraint_scores']
        return pass_rates

    def calculate_local_global_changes(self, constraint_pass_rates: Dict, valid_indices: Set[str]):
        local_change = {'local': {}, 'global': {}}
        global_change = {'local': {}, 'global': {}}
        
        for idx, pass_rates in constraint_pass_rates.items():
            if idx not in valid_indices:
                continue
                
            self._process_local_changes(idx, pass_rates, local_change)
            self._process_global_changes(idx, pass_rates, global_change)
        return local_change, global_change

    def _process_local_changes(self, idx: str, pass_rates: Dict, local_change: Dict):
        local_scores, global_scores = [], []
        
        for target_constraint, changed_constraints in pass_rates.items():
            if target_constraint in self.local_constraints and changed_constraints:
                self._calculate_scores(changed_constraints, local_scores, global_scores)
                
        if local_scores:
            local_change['local'][idx] = round(sum(local_scores) / len(local_scores), 2)
        if global_scores:
            local_change['global'][idx] = round(sum(global_scores) / len(global_scores), 2)

    def _process_global_changes(self, idx: str, pass_rates: Dict, global_change: Dict):
        local_scores, global_scores = [], []
        
        for target_constraint, changed_constraints in pass_rates.items():
            if target_constraint in self.global_constraints and changed_constraints:
                self._calculate_scores(changed_constraints, local_scores, global_scores)
                
        if local_scores:
            global_change['local'][idx] = round(sum(local_scores) / len(local_scores), 2)
        if global_scores:
            global_change['global'][idx] = round(sum(global_scores) / len(global_scores), 2)

    def _calculate_scores(self, constraints: Dict, local_scores: List, global_scores: List):
        for constraint_type, scores in constraints.items():
            if constraint_type in ['cuisine', 'room_rule', 'room_type']:
                if scores.get('is_correct') is not None:
                    local_scores.append(scores['is_correct'])
            elif constraint_type == 'budget':
                if scores.get('is_correct') is not None:
                    global_scores.append(scores['is_correct'])

class FullTurnEvaluator(ConstraintEvaluator):
    def calculate_full_turn_rates(self, results: Dict, valid_indices: Set[str]) -> tuple:
        local_rates = {}
        global_rates = {}
        error_sents = {}
        
        for idx, idx_results in results.items():
            if idx not in valid_indices:
                continue
                
            constraint_scores = idx_results[0]['detailed_results'][0]['constraint_scores']
            local_scores, global_scores = [], []
            
            for constraint, scores in constraint_scores.items():
                if constraint in ['cuisine', 'room_rule', 'room_type']:
                    if scores.get('is_correct') is not None:
                        local_scores.append(scores['is_correct'])
                elif constraint == 'budget':
                    if scores.get('is_correct') is not None:
                        global_scores.append(scores['is_correct'])
                elif "error:" in constraint_scores:
                    error_sents[idx] = constraint_scores
                    
            if local_scores:
                local_rates[idx] = sum(local_scores) / len(local_scores)
            if global_scores:
                global_rates[idx] = sum(global_scores) / len(global_scores)
                
        return local_rates, global_rates, error_sents

class ThreeTurnEvaluator(ConstraintEvaluator):
    def __init__(self, date: str, model: str, turn_type: str):
        super().__init__()
        self.date = date
        self.model = model
        self.turn_type = turn_type
        # self.turn_types = ['global_local', 'local_global']
        
    def load_three_turn_results(self) -> Dict:
        with open (f"/mnt/nas2/juhyun/FlexibleReasoningBench/implement/results/three_turn/results_{self.turn_type}_{self.date}_{self.model}.json", "r") as f:
            return json.load(f)

    def calculate_constraint_pass_rates(self, results: Dict) -> Dict:
        pass_rates = {}
        for idx, result in results.items():
            pass_rates[idx] = {
                'house_rule': None,
                'room_type': None,
                'cuisine': None,
                'budget': None,
            }
            for item in result:
                # Only consider the last turn's constraint scores
                import pdb; pdb.set_trace()
                constraint_type = item['constraint_type']
                constraint_type = next((c for c in item['constraint_type'] if c != 'budget'), None)
                constraint_type = constraint_type.replace(" ", "_")
                pass_rates[idx][constraint_type] = item['detailed_results'][-1]['constraint_scores']
                # pass_rates[idx][] = item['detailed_results'][-1]['constraint_scores']
        return pass_rates

    def calculate_local_changes(self, constraint_pass_rates: Dict, valid_indices: Set[str]):
        # global_local_change = {'local': {}, 'global': {}}
        local_change = {'local': {}, 'global': {}}
        
        for idx, pass_rates in constraint_pass_rates.items():
            if idx not in valid_indices:
                continue
                
            # self._process_global_local_changes(idx, pass_rates, global_local_change)
            self._process_local_changes(idx, pass_rates, local_change)
            # self._process_global_changes(idx, pass_rates, local_change)
        return local_change

    def _process_local_changes(self, idx: str, pass_rates: Dict, local_change: Dict):
        local_scores, global_scores = [], []
        
        for target_constraint, changed_constraints in pass_rates.items():
            if target_constraint in self.local_constraints and changed_constraints:
                self._calculate_scores(changed_constraints, local_scores, global_scores)
                
        if local_scores:
            local_change['local'][idx] = round(sum(local_scores) / len(local_scores), 2)
        if global_scores:
            local_change['global'][idx] = round(sum(global_scores) / len(global_scores), 2)

    def _process_global_changes(self, idx: str, pass_rates: Dict, global_change: Dict):
        local_scores, global_scores = [], []
        
        for target_constraint, changed_constraints in pass_rates.items():
            if target_constraint in self.global_constraints and changed_constraints:
                self._calculate_scores(changed_constraints, local_scores, global_scores)
                
        if local_scores:
            global_change['local'][idx] = round(sum(local_scores) / len(local_scores), 2)
        if global_scores:
            global_change['global'][idx] = round(sum(global_scores) / len(global_scores), 2)

    def _calculate_scores(self, constraints: Dict, local_scores: List, global_scores: List):
        for constraint_type, scores in constraints.items():
            if constraint_type in ['cuisine', 'room_rule', 'room_type']:
                if scores.get('is_correct') is not None:
                    local_scores.append(scores['is_correct'])
            elif constraint_type == 'budget':
                if scores.get('is_correct') is not None:
                    global_scores.append(scores['is_correct'])

def get_valid_indices(two_turn_results: Dict, full_turn_results: Dict) -> Set[str]:
    two_turn_indices = set()
    for idx, pass_rates in two_turn_results.items():
        has_valid_data = False
        for constraint_type, changed_constraints in pass_rates.items():
            if changed_constraints is not None:
                has_valid_data = True
                break
        if has_valid_data:
            two_turn_indices.add(idx)
    
    full_turn_indices = set(full_turn_results.keys())
    return two_turn_indices.intersection(full_turn_indices)

class TwoTurnPrefEvaluator(ConstraintEvaluator):
    def __init__(self, date: str, model: str, budget_size: str):
        super().__init__()
        self.date = date
        self.model = model
        self.budget_size = budget_size
        
    def load_two_turn_results(self) -> Dict:
        aggregated_results = {}
        for constraint in ['rating', 'cuisine']:
            file_path = f"/mnt/nas2/juhyun/FlexibleReasoningBench/implement/results/preference/results_{self.budget_size}_{constraint}_{self.date}_{self.model}_preference.json"
            results = self.load_json_file(file_path)
            for k, v in results.items():
                if v['constraint_type'] == constraint:
                    if k not in aggregated_results:
                        aggregated_results[k] = []
                    aggregated_results[k].append(v)
        return aggregated_results

    def calculate_constraint_pass_rates(self, results: Dict) -> Dict:
        pass_rates = {}
        for idx, result in results.items():
            pass_rates[idx] = {
                'cuisine': None,
                'rating': None
            }
            for item in result:
                # Only consider the last turn's constraint scores
                constraint_type = item['constraint_type']
                # constraint_type = constraint_type.replace(" ", "_")
                pass_rates[idx][constraint_type] = item['detailed_results'][-1]['constraint_scores']
        return pass_rates

    def calculate_preference_changes(self, constraint_pass_rates: Dict, valid_indices: Set[str]):
        # global_local_change = {'local': {}, 'global': {}}
        preference_change = {'local': {}, 'global': {}, 'preference': {}, 'preference_correct_global_incorrect': {'cuisine':[], 'rating':[]}, 'preference_incorrect_global_correct': {'cuisine':[], 'rating':[]}}
        
        for idx, pass_rates in constraint_pass_rates.items():
            if idx not in valid_indices:
                continue
            
            # self._process_global_local_changes(idx, pass_rates, global_local_change)
            self._process_preference_changes(idx, pass_rates, preference_change)
            # self._process_global_changes(idx, pass_rates, local_change)
            
            for constraint in pass_rates:
                if pass_rates[constraint]:
                    is_preference_correct = self._is_preference_correct(pass_rates[constraint])
                    is_global_correct = self._is_global_correct(pass_rates[constraint])
                    if is_preference_correct and not is_global_correct:
                        preference_change['preference_correct_global_incorrect'][constraint].append(idx)
                    elif not is_preference_correct and is_global_correct:
                        preference_change['preference_incorrect_global_correct'][constraint].append(idx)

        return preference_change

    def _process_preference_changes(self, idx: str, pass_rates: Dict, local_change: Dict):
        local_scores, global_scores, pref_scores = [], [], []
        
        for target_constraint, changed_constraints in pass_rates.items():
            if target_constraint in ['rating', 'cuisine'] and changed_constraints:
                self._calculate_scores(changed_constraints, local_scores, global_scores, pref_scores)
                
        if local_scores:
            local_change['local'][idx] = round(sum(local_scores) / len(local_scores), 2)
        if global_scores:
            local_change['global'][idx] = round(sum(global_scores) / len(global_scores), 2)
        if pref_scores:
            local_change['preference'][idx] = round(sum(pref_scores) / len(pref_scores), 2)
    
    def _calculate_scores(self, constraints: Dict, local_scores: List, global_scores: List, pref_scores: List):
        for constraint_type, scores in constraints.items():
            if constraint_type in ['cuisine', 'room_rule', 'room_type']:
                if scores.get('is_correct') is not None:
                    local_scores.append(scores['is_correct'])
            elif constraint_type == 'budget':
                if scores.get('is_correct') is not None:
                    global_scores.append(scores['is_correct'])
            elif constraint_type in ['rating_pref', 'cuisine_pref']:
                if scores.get('is_correct') is not None:
                    pref_scores.append(scores['is_correct'])


    def _is_preference_correct(self, constraint_scores: Dict) -> bool:
        """
        Check if all preference constraints are correct.
        """
        for constraint in self.preference_constraints:
            if constraint_scores.get(constraint) and constraint_scores[constraint].get('is_correct') is False:
                return False
        return True

    def _is_global_correct(self, constraint_scores: Dict) -> bool:
        """
        Check if all global constraints are correct.
        """
        for constraint in self.global_constraints:
            if constraint_scores.get(constraint) and constraint_scores[constraint].get('is_correct') is False:
                return False
        return True


def main():
    # Load all results first
    two_turn = TwoTurnEvaluator("20250126", "gpt_reevaluated")
    two_turn_results = two_turn.load_two_turn_results()
    constraint_rates = two_turn.calculate_constraint_pass_rates(two_turn_results)
    # for budget_size in ['small', 'middle', 'high']:
    #     two_turn_pref = TwoTurnPrefEvaluator("20250130", "llama", budget_size)
    #     two_turn_pref_results = two_turn_pref.load_two_turn_results()
    #     constraint_pref_rates = two_turn_pref.calculate_constraint_pass_rates(two_turn_pref_results)
    #     valid_indices_pref = get_valid_indices(constraint_pref_rates, two_turn_pref_results)

    #     # Calculate preference changes
    #     preference_changes = two_turn_pref.calculate_preference_changes(constraint_pref_rates, valid_indices_pref)
    #     import pdb; pdb.set_trace()
    #     preference_correct_global_incorrect = len(preference_changes['preference_correct_global_incorrect'])
    #     preference_incorrect_global_correct = len(preference_changes['preference_incorrect_global_correct'])

    #     # Calculate average pass rates for preference constraints
    #     local_pref_avg_pass_rate = two_turn_pref.calculate_average_pass_rates(preference_changes['local'])
    #     global_pref_avg_pass_rate = two_turn_pref.calculate_average_pass_rates(preference_changes['global'])
    #     pref_avg_pass_rate = two_turn_pref.calculate_average_pass_rates(preference_changes['preference'])

    #     print("Local Preference Average Pass Rate:", local_pref_avg_pass_rate)
    #     print("Global Preference Average Pass Rate:", global_pref_avg_pass_rate)
    #     print("Preference Average Pass Rate:", pref_avg_pass_rate)

    #     print("Preference Correct Global Incorrect:", preference_correct_global_incorrect)
    #     print("Preference Incorrect Global Correct:", preference_incorrect_global_correct)
    full_turn = FullTurnEvaluator()
    full_results = full_turn.load_aggregated_results("/mnt/nas2/juhyun/FlexibleReasoningBench/implement/results/results_all_at_once_20250129_llama_not_easy.json")
    
    # Get valid indices that exist in both datasets
    valid_indices = get_valid_indices(constraint_rates, full_results)
    # missing indices = set(constraint_rates.keys()) - valid_indices
    # [63, 86, 98, 113, 169, 178]
    print(f"\nNumber of valid data points: {len(valid_indices)}")
    
    # # Calculate rates using only valid indices
    local_changes, global_changes = two_turn.calculate_local_global_changes(constraint_rates, valid_indices)
    # global_keys = {v for v in global_changes['local'].keys()}
    # missing_indices = valid_indices - global_keys
    local_rates, global_rates, errors = full_turn.calculate_full_turn_rates(full_results, valid_indices)
    
    # Calculate averages
    local_change_local_avg = two_turn.calculate_average_pass_rates(local_changes['local'])
    local_change_global_avg = two_turn.calculate_average_pass_rates(local_changes['global'])
    global_change_local_avg = two_turn.calculate_average_pass_rates(global_changes['local'])
    global_change_global_avg = two_turn.calculate_average_pass_rates(global_changes['global'])
    
    full_local_avg = full_turn.calculate_average_pass_rates(local_rates)
    full_global_avg = full_turn.calculate_average_pass_rates(global_rates)

    # three_turn_gl = ThreeTurnEvaluator(date="20250130", model="gpt", turn_type="global_local")
    # three_turn_lg = ThreeTurnEvaluator(date="20250130", model="gpt", turn_type="local_global")

    # # Load three turn results
    # three_turn_gl_results = three_turn_gl.load_three_turn_results()
    # three_turn_gl_constraint_rates = three_turn_gl.calculate_constraint_pass_rates(three_turn_gl_results)

    # # Get valid indices that exist in both datasets
    # three_turn_valid_indices = get_valid_indices(three_turn_gl_constraint_rates, full_results)

    # # Calculate rates using only valid indices
    # three_turn_local_changes_gl = three_turn_gl.calculate_local_changes(three_turn_gl_constraint_rates, three_turn_valid_indices)
    # # Calculate averages
    # three_turn_gl_local_avg = three_turn_gl.calculate_average_pass_rates(three_turn_local_changes_gl['local'])
    # three_turn_gl_global_avg = three_turn_gl.calculate_average_pass_rates(three_turn_local_changes_gl['global'])
    
    # three_turn_lg_results = three_turn_lg.load_three_turn_results()
    # three_turn_lg_constraint_rates = three_turn_lg.calculate_constraint_pass_rates(three_turn_lg_results)
    # three_turn_lg_valid_indices = get_valid_indices(three_turn_lg_constraint_rates, full_results)
    # three_turn_local_changes = three_turn_lg.calculate_local_changes(three_turn_lg_constraint_rates, three_turn_lg_valid_indices)
    # three_turn_lg_local_avg = three_turn_lg.calculate_average_pass_rates(three_turn_local_changes['local'])
    # three_turn_lg_global_avg = three_turn_lg.calculate_average_pass_rates(three_turn_local_changes['global'])

    # # Print results
    print(f"\nTwo-turn Results:")
    print(f"Local Change - Local Constraints: {local_change_local_avg}")
    print(f"Local Change - Global Constraints: {local_change_global_avg}")
    print(f"Global Change - Local Constraints: {global_change_local_avg}")
    print(f"Global Change - Global Constraints: {global_change_global_avg}")
    
    # print(f"\nFull-turn Results:")
    # print(f"Local Constraints: {full_local_avg}")
    # print(f"Global Constraints: {full_global_avg}")

    # print(f"\nThree-turn Results:")
    # print(f"Global_Local Change - Local Constraints: {three_turn_gl_local_avg}")
    # print(f"Global_Local Change - Global Constraints: {three_turn_gl_global_avg}")
    # print(f"Local_Global Change - Local Constraints: {three_turn_lg_local_avg}")
    # print(f"Local_Global Change - Global Constraints: {three_turn_lg_global_avg}")
    
    # # Print number of data points for each category
    # print(f"\nDetailed data points:")
    # print(f"Local changes - Local constraints: {len(local_changes['local'])}")
    # print(f"Local changes - Global constraints: {len(local_changes['global'])}")
    # print(f"Global changes - Local constraints: {len(global_changes['local'])}")
    # print(f"Global changes - Global constraints: {len(global_changes['global'])}")
    # print(f"Full turn - Local constraints: {len(local_rates)}")
    # print(f"Full turn - Global constraints: {len(global_rates)}")
    # print(f"Three turn - Global_Local - Local constraints: {len(three_turn_local_changes['local'])}")
    # print(f"Three turn - Global_Local - Global constraints: {len(three_turn_local_changes['global'])}")
    # print(f"Three turn - Local_Global - Local constraints: {len(three_turn_local_changes['local'])}")
    # print(f"Three turn - Local_Global - Global constraints: {len(three_turn_local_changes['global'])}")

if __name__ == "__main__":
    main()

