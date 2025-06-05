<div align="center">
  <h1>Flex-TravelPlanner: A Benchmark for Flexible Planning with Language Agents</h1>
  <p>
    <a href="https://openreview.net/forum?id=a7unQ5jMx7"><img src="https://img.shields.io/badge/Paper-FlexTravelPlanner-green" alt="Paper"></a>
    <a href="https://github.com/juhyunohh/FlexTravelBench/"><img src="https://img.shields.io/badge/Github-FlexTravelPlanner-blue" alt="Code"></a>
  </p>
</div>  

This is the official GitHub repository for [Flex-TravelPlanner: A Benchmark for Flexible Planning with Language Agents]([https://arxiv.org/abs/2412.10424](https://openreview.net/forum?id=a7unQ5jMx7)).


# Dataset Generation for Evaluation

## Database

Files in the `./agents/evaluation/database` directory:
- `{test|train|validation}_ref_info.jsonl`: reference information for scoring
- `val_dataset_without_{constraint_type}_one.json`: dataset with one constraint removed

---

## Dataset Generation Script (`dataset_generate.py`)

This script generates evaluation datasets by removing constraints from the original TravelPlanner dataset.  
There are two main generation strategies:

1. Removing single constraints
2. Removing pairs of constraints (global-local combinations)

### Usage Example

```python
from dataset_generate import generate_dataset

# Generate test datasets with single constraints removed
generate_dataset(
    save_path="./evaluation/database",
    single_constraints=['budget', 'house rule', 'room type', 'cuisine']
)

# Generate test datasets with constraint pairs removed
generate_dataset(
    save_path="./evaluation/database", 
    global_constraints=['budget', 'people_number'],
    local_constraints=['house rule', 'room type', 'cuisine']
)
```

# Evaluation Script

## Overview
After generating the evaluation datasets, use this script (`evaluate.py`) to run evaluations in different modes.

## Usage

### 1. Single Constraint Mode
Evaluates the model's performance when a single constraint is removed.

```bash
python evaluate.py --mode single_constraint \
                  --constraints "budget,room type" \
                  --output_dir "./results/two_turn"
```

### 2. Two Constraints Mode
Evaluates how the model handles cases where pairs of constraints are removed.

```bash
python evaluate.py --mode two_constraints \
                  --constraint_pairs "global_local,local_global" \
                  --output_dir "./results/three_turn"
```                  
### 3. All-at-once Mode
Evaluates the model on all examples with specified difficulty levels.

Exclude easy examples

```bash
python evaluate.py --mode all_at_once \
                  --difficulty not_easy \
                  --output_dir "./results/one_turn"
```

### 4. Preference Mode
Evaluates the model's handling of preference-based constraints across different budget types.

```bash
python evaluate.py --mode preference \
                  --budget_types "high,middle,small" \
                  --preference_types "cuisine,rating" \
                  --output_dir "./results/preference"
```
---

# config/test.yaml

- The `history` option:
  - `1` keeps track of all previous logs interactively.
  - `0` summarizes instead of storing the full log.
