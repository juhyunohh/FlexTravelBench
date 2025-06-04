import re
import json
from prompts import PLAN_PARSING
from tqdm import tqdm
import pandas as pd
from utils import inference_gpt, MODEL_PATHS


def build_plan_format_conversion_prompt(plan_text):
    prompt = PLAN_PARSING + "Text:\n" + plan_text
    return prompt


def parse_plan(plan_text):
    """
    Parses the textual travel plan into a structured format for evaluation.
    """
    # extracted_plan_text = extract_revised_plan_section(plan_text)
    prompt_for_parsing = build_plan_format_conversion_prompt(plan_text)
    parsed_plan_text = inference_gpt(MODEL_PATHS["gpt-4o"], prompt_for_parsing)
    try:
        pattern = r"json(.*?)```"
        parsed_plan_text = re.search(pattern, parsed_plan_text, re.DOTALL).group(1).strip()
        cleaned_text = parsed_plan_text.replace("{{", "{").replace("}}", "}")
        output = json.loads(cleaned_text.replace("\n", "").replace("```json","").replace("```", ""))
    except:
        output = []
    return output

def convert_to_valid_json(invalid_json_str):
    valid_json_str = re.sub(r"'", '"', invalid_json_str)  # Replace single quotes with double quotes
    valid_json_str = re.sub(r"\bNone\b", "null", valid_json_str)  # Replace Python None with JSON null
    return valid_json_str

# def process_csv(input_csv_path, output_json_path, constraint):
#     """
#     Processes the input CSV and generates an evaluation JSON file.
#     """
#     # Read the CSV file
#     df = pd.read_csv(input_csv_path)

#     # Combine query_data and tested_data
#     evaluation_data = []
#     for _, row in tqdm(df.iterrows(), total=len(df)):
#         # Parse query_data
#         # query_data = generate_query_data(row, constraint)

#         # Parse tested_data from new_plan
#         tested_data = parse_plan(row["new_plan"])

#         # Combine into evaluation structure
#         evaluation_entry = {
#             "query_data": query_data,
#             "tested_data": tested_data
#         }
#         evaluation_data.append(evaluation_entry)
    
#     # Save to JSON file
#     with open(output_json_path, "w") as f:
#         json.dump(evaluation_data, f, indent=4)

#     print(f"Evaluation file has been generated at: {output_json_path}")
