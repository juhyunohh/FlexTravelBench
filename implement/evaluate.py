import click
from dotenv import load_dotenv
from agents import Evaluatee, Evaluator, Runner
from agents.utils import console, load_config, setup_logging, load_line_json_data
from rich.panel import Panel
import random
import json
import numpy as np
from datetime import datetime
import os
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset

@click.command()
@click.option(
    "--config", "-c",
    default="configs/test.yaml",
    help="Path to the configuration file",
    type=click.Path(exists=True)
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show all conversation utterances"
)
@click.option(
    "--mode",
    type=click.Choice(["single_constraint", "two_constraints", "all_at_once", "preference"]),
    default="single_constraint",
    help="Mode to run: single_constraint, two_constraints, all_at_once, or preference"
)
@click.option(
    "--ref_file",
    default="./agents/evaluation/database/validation_ref_info.jsonl",
    help="Path to reference information file",
    type=click.Path(exists=True)
)
@click.option(
    "--output_dir",
    default="./results",
    help="Directory to save results",
    type=click.Path()
)
@click.option(
    "--dataset_dir",
    default="./agents/evaluation/database",
    help="Directory containing dataset files",
    type=click.Path(exists=True)
)
@click.option(
    "--constraints",
    default=None,
    help="Comma-separated list of constraints to evaluate (for single_constraint mode)"
)
@click.option(
    "--constraint_pairs",
    default=None,
    help="Comma-separated list of constraint pairs (format: first_second,third_fourth) for two_constraints mode"
)
@click.option(
    "--difficulty",
    type=click.Choice(["all", "easy", "not_easy", "medium", "hard"]),
    default="all",
    help="Filter dataset by difficulty level (for all_at_once mode)"
)

@click.option(
    "--budget_types",
    default="high,middle,small",
    help="Comma-separated list of budget types for preference mode"
)
@click.option(
    "--preference_types",
    default="cuisine,rating",
    help="Comma-separated list of preferences to evaluate"
)

def main(
    config: str,
    verbose: bool,
    mode: str,
    ref_file: str,
    output_dir: str,
    dataset_dir: str,
    constraints: Optional[str],
    constraint_pairs: Optional[str],
    difficulty: str,
    budget_types: str,
    preference_types: str
):
    """Run an automated interview evaluation system with flexible constraints."""
    # Show welcome message
    console.print(
        Panel(
            "[green]Automated Interview Evaluation System[/green]\n"
            "[cyan]Use Ctrl+C to exit at any time[/cyan]",
            border_style="green",
            padding=(1, 2),
        )
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    config_data = load_config(config)

    if mode == "preference":
        run_preference_mode(
            config_data=config_data,
            verbose=verbose,
            ref_file=ref_file,
            output_dir=output_dir,
            dataset_dir=dataset_dir,
            budget_types=budget_types.split(","),
            preference_types=preference_types.split(",")
        )

    elif mode == "single_constraint":
        run_single_constraint_mode(
            config_data=config_data,
            verbose=verbose,
            ref_file=ref_file,
            output_dir=output_dir,
            dataset_dir=dataset_dir,
            constraints=constraints.split(",") if constraints else None
        )
    elif mode == "two_constraints":
        run_two_constraints_mode(
            config_data=config_data,
            verbose=verbose,
            ref_file=ref_file,
            output_dir=output_dir,
            dataset_dir=dataset_dir,
            constraint_pairs=parse_constraint_pairs(constraint_pairs) if constraint_pairs else None
        )
    else:  # all_at_once mode
        run_all_at_once_mode(
            config_data=config_data,
            verbose=verbose,
            ref_file=ref_file,
            output_dir=output_dir,
            difficulty=difficulty
        )

def parse_constraint_pairs(constraint_pairs_str: str) -> List[Tuple[str, str]]:
    """Parse constraint pairs from string format 'first_second,third_fourth'"""
    pairs = []
    for pair_str in constraint_pairs_str.split(","):
        pair = tuple(pair_str.split("_"))
        if len(pair) == 2:
            pairs.append(pair)
        else:
            console.print(f"[yellow]Warning: Invalid constraint pair format: {pair_str}[/yellow]")
    return pairs

def run_single_constraint_mode(
    config_data: Dict,
    verbose: bool,
    ref_file: str,
    output_dir: str,
    dataset_dir: str,
    constraints: Optional[List[str]] = None
):
    """Run evaluation with single constraints."""
    # Default constraints if none provided
    if not constraints:
        constraints = ['budget', 'house rule', 'room type', 'cuisine', 'people_number']
    
    console.print(f"[blue]Running single constraint mode with constraints: {constraints}[/blue]")
    
    for constraint in constraints:
        all_results = {}
        normalized_constraint = constraint.replace(" ", "_")
        dataset_file = f"{dataset_dir}/dataset_without_{normalized_constraint}_one.json"
        
        timestamp = datetime.now().strftime("%Y%m%d")
        results_file = os.path.join(output_dir, f"results_{normalized_constraint}_{timestamp}_gpt.json")
        
        try:
            with open(dataset_file, "r") as f:
                dataset = json.load(f)

            console.print(f"[blue]Processing constraint: {constraint} with {len(dataset)} data points[/blue]")

            for data in dataset:
                process_data_item(
                    data=data,
                    ref_file=ref_file,
                    config_data=config_data,
                    verbose=verbose,
                    all_results=all_results,
                    constraint_type=normalized_constraint
                )
                
            # Save final results for this constraint
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=4)
                
            console.print(f"[green]Results for constraint {constraint} saved to {results_file}[/green]")
                
        except Exception as e:
            console.print(f"\n[red]Error in processing dataset {dataset_file}: {str(e)}[/red]")
            continue
    
    console.print("\n[green]All single constraints processed successfully![/green]")

def run_two_constraints_mode(
    config_data: Dict,
    verbose: bool,
    ref_file: str,
    output_dir: str,
    dataset_dir: str,
    constraint_pairs: Optional[List[Tuple[str, str]]] = None
):
    """Run evaluation with two constraints."""
    # Default constraint conditions if none provided
    if not constraint_pairs:
        constraint_conditions = ['global_local', 'local_global']
    else:
        constraint_conditions = [f"{pair[0]}_{pair[1]}" for pair in constraint_pairs]
    
    console.print(f"[blue]Running two constraints mode with conditions: {constraint_conditions}[/blue]")
    
    for condition in constraint_conditions:
        results_file = os.path.join(output_dir, f"results_{condition}_{datetime.now().strftime('%Y%m%d')}_gpt.json")
        
        # Load existing results if file exists
        all_results = {}
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                all_results = json.load(f)
        
        dataset_file = f"{dataset_dir}/{condition}_dataset.json"
        
        try:
            with open(dataset_file, "r") as f:
                dataset = json.load(f)
            
            # Filter dataset based on condition
            new_dataset = filter_dataset_by_condition(dataset, condition)
            
            console.print(f"[blue]Processing condition: {condition} with {len(new_dataset)} data points[/blue]")
            
            for data in new_dataset:
                idx_str = str(data['idx'])
                current_constraints = [key for item in data['new_constraints'] for key in item]
                
                # Skip if this data item has already been processed
                if is_already_processed(idx_str, current_constraints, all_results):
                    continue
                
                process_data_item(
                    data=data,
                    ref_file=ref_file,
                    config_data=config_data,
                    verbose=verbose,
                    all_results=all_results,
                    constraint_type=None,  # Will be set from data['new_constraints']
                    is_two_constraints=True
                )
                
                # Save updated results after each data item
                with open(results_file, 'w') as f:
                    json.dump(all_results, f, indent=4)
                
                console.print(f"[green]Saved results for data {idx_str}[/green]")
            
            console.print(f"\n[green]Done with condition {condition}![/green]")
            
        except Exception as e:
            console.print(f"\n[red]Error processing condition {condition}: {str(e)}[/red]")

def run_all_at_once_mode(
    config_data: Dict,
    verbose: bool,
    ref_file: str,
    output_dir: str,
    difficulty: str = "all"
):
    """Run evaluation on all examples at once without adding constraints."""
    console.print(f"[blue]Running all-at-once mode with difficulty filter: {difficulty}[/blue]")
    
    all_results = {}
    
    # Load dataset directly using the datasets library
    dataset = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
    
    timestamp = datetime.now().strftime("%Y%m%d")
    results_file = os.path.join(output_dir, f"results_all_at_once_{timestamp}_gpt_{difficulty}.json")
    
    # Count total examples to process
    total_examples = 0
    for idx, data in enumerate(dataset):
        if should_process_example(data, difficulty):
            total_examples += 1
    
    console.print(f"[blue]Processing {total_examples} examples with difficulty: {difficulty}[/blue]")
    
    processed_count = 0
    for idx, data in enumerate(dataset):
        if should_process_example(data, difficulty):
            processed_count += 1
            console.print(f"[cyan]Processing example {processed_count}/{total_examples} (dataset index: {idx})[/cyan]")
            
            try:
                seed_question = data['query'] 
                ref_data = load_line_json_data(ref_file)[idx]
                
                exclude_keys = {'query', 'level', 'annotated_plan', 'reference_information'}
                initial_constraints = {k: data[k] for k in data.keys() - exclude_keys}
                
                # No new constraints for this mode
                new_constraints = []
                
                evaluator = Evaluator(config=config_data, name="Evaluator")
                student = Evaluatee(config=config_data, name="Evaluatee")

                logger, log_file_path = setup_logging(config_data, verbose)
                evaluator.seed_question = seed_question
                evaluator.constraints_dict = initial_constraints
                evaluator.new_constraints = new_constraints
                evaluator.ref_data = ref_data
                runner = Runner(evaluator, student, config_data, logger, log_file_path, console)
                
                runner.run()
                
                result_entry = {
                    'data_idx': idx,
                    'difficulty': data['level'],
                    'constraint_type': None,
                    'detailed_results': [
                        {
                            'response': runner.responses[0],
                            'constraint_scores': runner.scores['total_constraints_score'],
                            'total_pass_rate': round(runner.scores['total_pass_rate'], 2)
                        }
                    ],
                }
                
                # Add to results
                all_results[str(idx)] = result_entry
                
                # Save results after each example to avoid losing progress
                if processed_count % 5 == 0:  # Save every 5 examples
                    with open(results_file, 'w') as f:
                        json.dump(all_results, f, indent=4)
                    console.print(f"[green]Intermediate results saved to {results_file}[/green]")
                
            except Exception as e:
                console.print(f"\n[red]Error in processing data {idx}: {str(e)}[/red]")
                all_results[str(idx)] = {
                    'data_idx': idx,
                    'difficulty': data.get('level', 'unknown'),
                    'constraint_type': None,
                    'detailed_results': [
                        {
                            'response': None,
                            'constraint_scores': f"error: {str(e)}",
                            'total_pass_rate': None
                        }
                    ],
                }
                continue
    
    # Save final results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    console.print(f"\n[bold green]All done! Results saved to {results_file}[/bold green]")

def run_preference_mode(
    config_data: Dict,
    verbose: bool,
    ref_file: str,
    output_dir: str,
    dataset_dir: str,
    budget_types: List[str],
    preference_types: List[str]
):
    """Run evaluation for preference-based constraints."""
    console.print(f"[blue]Running preference mode with budget types: {budget_types} and preferences: {preference_types}[/blue]")
    
    for budget_type in budget_types:
        for preference in preference_types:
            timestamp = datetime.now().strftime("%Y%m%d")
            results_file = os.path.join(output_dir, f"results_{budget_type}_{preference}_{timestamp}_preference.json")
            
            # Load existing results if file exists
            all_results = {}
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    all_results = json.load(f)

            dataset_file = os.path.join(dataset_dir, f"preference/val_dataset_full_{budget_type}_budget.json")
            
            try:
                with open(dataset_file, "r") as f:
                    dataset = json.load(f)

                for data in dataset:
                    process_preference_data_item(
                        data=data,
                        preference_type=preference,
                        ref_file=ref_file,
                        config_data=config_data,
                        verbose=verbose,
                        all_results=all_results,
                        results_file=results_file
                    )
                    
            except Exception as e:
                console.print(f"\n[red]Error in processing dataset {dataset_file}: {str(e)}[/red]")
                continue
    
    console.print("\n[green]All preference evaluations completed![/green]")

def process_preference_data_item(
    data: Dict,
    preference_type: str,
    ref_file: str,
    config_data: Dict,
    verbose: bool,
    all_results: Dict,
    results_file: str
):
    """Process a single preference data item."""
    idx_str = str(data['idx'])
    
    # Skip if already processed
    if idx_str in all_results:
        console.print(f"\n[yellow]Skipping already processed data {idx_str}[/yellow]")
        return

    try:
        seed_question = data['query']
        ref_data = load_line_json_data(ref_file)[data['idx']]
        preference_constraints = data['preference_constraint']
        pref_constraint = preference_constraints[preference_type]

        if pref_constraint is None:
            return

        new_constraints = [{f'{preference_type}_pref': pref_constraint}]
        exclude_keys = {'query', 'level', 'annotated_plan', 'reference_information', 
                       'new_constraints', 'preference_constraint'}
        initial_constraints = {k: data[k] for k in data.keys() - exclude_keys}

        # Run evaluation
        evaluator = Evaluator(config=config_data, name="Evaluator")
        student = Evaluatee(config=config_data, name="Evaluatee")
        
        logger, log_file_path = setup_logging(config_data, verbose)
        evaluator.seed_question = seed_question
        evaluator.constraints_dict = initial_constraints
        evaluator.new_constraints = new_constraints
        evaluator.ref_data = ref_data
        runner = Runner(evaluator, student, config_data, logger, log_file_path, console)
        
        runner.run()

        # Store results
        result_entry = {
            'constraint_type': preference_type,
            'data_idx': idx_str,
            'detailed_results': [
                {
                    "turn": i+1,
                    "response": runner.responses[i],
                    "constraint_scores": runner.scores['all_total_constraints_score'][i],
                    "total_pass_rate": runner.scores['total_pass_rate']
                } for i in range(len(runner.responses))
            ]
        }

        all_results[idx_str] = result_entry

        # Save results
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)

    except Exception as e:
        console.print(f"\n[red]Error in processing data {idx_str}: {str(e)}[/red]")
        all_results[idx_str] = {
            'data_idx': idx_str,
            'constraint_type': preference_type,
            'detailed_results': [
                {
                    'response': None,
                    'constraint_scores': f"error: {str(e)}",
                    'total_pass_rate': None
                }
            ]
        }
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)

def should_process_example(data, difficulty):
    """Determine if an example should be processed based on difficulty filter."""
    if difficulty == "all":
        return True
    elif difficulty == "not_easy":
        return data['level'] != 'easy'
    else:
        return data['level'] == difficulty

def filter_dataset_by_condition(dataset, condition):
    """Filter dataset based on constraint condition."""
    new_dataset = []
    
    if condition == 'global_local':
        new_dataset = [data for data in dataset if data['new_constraints'][0].get('budget') and data['new_constraints'][1].get('room type')]
        new_dataset += [data for data in dataset if data['new_constraints'][0].get('budget') and data['new_constraints'][1].get('house rule')]
        new_dataset += [data for data in dataset if data['new_constraints'][0].get('budget') and data['new_constraints'][1].get('cuisine')]
    elif condition == 'local_global':
        new_dataset = [data for data in dataset if data['new_constraints'][0].get('room type') and data['new_constraints'][1].get('budget')]
        new_dataset += [data for data in dataset if data['new_constraints'][0].get('house rule') and data['new_constraints'][1].get('budget')]
        new_dataset += [data for data in dataset if data['new_constraints'][0].get('cuisine') and data['new_constraints'][1].get('budget')]
    else:
        # For custom constraint pairs, parse the condition
        parts = condition.split('_')
        if len(parts) == 2:
            first, second = parts
            new_dataset = [data for data in dataset if 
                          (data['new_constraints'][0].get(first) and data['new_constraints'][1].get(second))]
    
    return new_dataset

def is_already_processed(idx_str, current_constraints, all_results):
    """Check if a data item has already been processed."""
    if idx_str in all_results:
        for result in all_results[idx_str]:
            if set(result['constraint_type']) == set(current_constraints):
                console.print(f"\n[yellow]Skipping already processed data {idx_str} with constraints {current_constraints}[/yellow]")
                return True
    return False

def process_data_item(
    data: Dict,
    ref_file: str,
    config_data: Dict,
    verbose: bool,
    all_results: Dict,
    constraint_type: Optional[str] = None,
    is_two_constraints: bool = False
):
    """Process a single data item and store results."""
    idx_str = str(data['idx']) if 'idx' in data else str(data.get('data_idx', 0))
    
    try:
        seed_question = data['query']
        ref_data = load_line_json_data(ref_file)[data['idx'] if 'idx' in data else data.get('data_idx', 0)]
        new_constraints = data.get('new_constraints', [])
        exclude_keys = {'query', 'level', 'annotated_plan', 'reference_information', 'new_constraints', 'idx', 'data_idx'}
        initial_constraints = {k: data[k] for k in data.keys() - exclude_keys}
        
        # Initialize evaluator and student
        evaluator = Evaluator(config=config_data, name="Evaluator")
        student = Evaluatee(config=config_data, name="Evaluatee")

        logger, log_file_path = setup_logging(config_data, verbose)
        evaluator.seed_question = seed_question
        evaluator.constraints_dict = initial_constraints
        evaluator.new_constraints = new_constraints
        evaluator.ref_data = ref_data
        runner = Runner(evaluator, student, config_data, logger, log_file_path, console)
        
        # Run the evaluation
        runner.run()
        
        # Prepare results
        if is_two_constraints:
            result_entry = {
                'constraint_type': [key for item in new_constraints for key in item],
                'data_idx': data['idx'] if 'idx' in data else data.get('data_idx', 0),
                'detailed_results': [
                    { 
                        "turn": i+1, 
                        "response": runner.responses[i], 
                        "constraint_scores": runner.scores['all_total_constraints_score'][i],
                        "total_pass_rate": runner.scores['total_pass_rate'] 
                    } for i in range(len(runner.responses))
                ]
            }
            
            # Add to results
            if idx_str not in all_results:
                all_results[idx_str] = []
            all_results[idx_str].append(result_entry)
        else:
            # For single constraint
            result_entry = {
                'constraint_type': constraint_type,
                'data_idx': data['idx'] if 'idx' in data else data.get('data_idx', 0),
                'detailed_results': [
                    {
                        "turn": i+1,
                        "response": runner.responses[i],
                        "constraint_scores": runner.scores['all_total_constraints_score'][i] if 'all_total_constraints_score' in runner.scores else runner.scores['total_constraints_score'],
                        "total_pass_rate": runner.scores['total_pass_rate']
                    } for i in range(len(runner.responses))
                ]
            }
            all_results[idx_str] = result_entry
        
    except Exception as e:
        console.print(f"\n[red]Error in processing data {idx_str}: {str(e)}[/red]")
        
        # Store error information
        error_entry = {
            'data_idx': data['idx'] if 'idx' in data else data.get('data_idx', 0),
            'constraint_type': constraint_type if not is_two_constraints else [key for item in data.get('new_constraints', []) for key in item],
            'detailed_results': [
                {
                    'response': None,
                    'constraint_scores': f"error: {str(e)}",
                    'total_pass_rate': None
                }
            ],
        }
        
        if is_two_constraints:
            if idx_str not in all_results:
                all_results[idx_str] = []
            all_results[idx_str].append(error_entry)
        else:
            all_results[idx_str] = error_entry

if __name__ == "__main__":
    main()
