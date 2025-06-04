import json
from typing import Dict, List, Optional, Set
from collections import defaultdict


class ConstraintEvaluator:
    def __init__(self):
        self.constraints = ['house_rule', 'room_type', 'cuisine', 'budget']
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

class FullTurnEvaluator(ConstraintEvaluator):
    def calculate_full_turn_rates(self, results: Dict) -> tuple:
        house_rule_rates = {}
        room_type_rates = {}
        cuisine_rates = {}
        budget_rates = {}
        error_sents = {}
        
        for idx, idx_results in results.items():
            # if idx not in valid_indices:
            #     continue
                
            constraint_scores = idx_results[0]['detailed_results'][0]['constraint_scores']
            house_rule_scores = []
            room_type_scores = []
            cuisine_scores = []
            budget_scores = []
            
            for constraint, scores in constraint_scores.items():
                if constraint == 'budget':
                    if scores.get('is_correct') is not None:
                        budget_scores.append(scores['is_correct'])
                elif constraint == 'cuisine':
                    if scores.get('is_correct') is not None:
                        cuisine_scores.append(scores['is_correct'])
                elif constraint == 'room_rule':
                    if scores.get('is_correct') is not None:
                        house_rule_scores.append(scores['is_correct'])
                elif constraint == 'room_type':
                    if scores.get('is_correct') is not None:
                        room_type_scores.append(scores['is_correct'])
                elif "error:" in constraint_scores:
                    error_sents[idx] = constraint_scores
                    
            # if local_scores:
            #     local_rates[idx] = sum(local_scores) / len(local_scores)
            # if global_scores:
            #     global_rates[idx] = sum(global_scores) / len(global_scores)
            if house_rule_scores:
                house_rule_rates[idx] = sum(house_rule_scores) / len(house_rule_scores)
            if room_type_scores:
                room_type_rates[idx] = sum(room_type_scores) / len(room_type_scores)
            if cuisine_scores:
                cuisine_rates[idx] = sum(cuisine_scores) / len(cuisine_scores)
            if budget_scores:
                budget_rates[idx] = sum(budget_scores) / len(budget_scores)
                
        return house_rule_rates, room_type_rates, cuisine_rates, budget_rates, error_sents
    
    def calculate_full_turn_pass_rates(self, results: Dict) -> Dict:
        house_rule_rates, room_type_rates, cuisine_rates, budget_rates, error_sents = self.calculate_full_turn_rates(results)
        import pdb; pdb.set_trace()
        house_rule_pass_rate = self.calculate_average_pass_rates(house_rule_rates)
        room_type_pass_rate = self.calculate_average_pass_rates(room_type_rates)
        cuisine_pass_rate = self.calculate_average_pass_rates(cuisine_rates)
        budget_pass_rate = self.calculate_average_pass_rates(budget_rates)

        return {
            'house_rule': house_rule_pass_rate,
            'room_type': room_type_pass_rate,
            'cuisine': cuisine_pass_rate,
            'budget': budget_pass_rate,
            'error_sents': error_sents
        }


def main():
    pass_rate_calculator = FullTurnEvaluator()
    # results = pass_rate_calculator.load_aggregated_results("results.json")
    results = pass_rate_calculator.load_aggregated_results("/mnt/nas2/juhyun/FlexibleReasoningBench/implement/results/results_all_at_once_20250129_llama_not_easy.json")
    pass_rates = pass_rate_calculator.calculate_full_turn_pass_rates(results)
    print(pass_rates)

if __name__ == "__main__":
    main()
