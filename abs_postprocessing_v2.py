import json
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, help='Path to the input JSONL file')
parser.add_argument('--output_file', type=str, help='Path to the output JSONL file')

input_file = parser.parse_args().input_file
output_file = parser.parse_args().output_file

with open(input_file, 'r') as f:
    data = [json.loads(line) for line in f]

with open(output_file, 'w') as f:
    for item in data:
        abstract = item['prediction']
        if "**Conditions:**" in abstract:
            abstract = "**Conditions:**" + abstract.split("**Conditions:**")[1]
            if "```" in abstract:
                abstract = abstract.split("```")[0].strip()
        # pre_abstract = item['question'].split("\n\nCandidate Abstract:\n")[1].split("\n```\n\n---\n\nPlease provide your output strictly following")[0]
        # if abstract == pre_abstract:
        #     pass
        # else:
        f.write(json.dumps(
            {
                "question": item["question"].split("\n\nInput to Process:\n```\nOriginal Query:\n")[1].split("\n\nCandidate Abstract:\n")[0],
                "abstract": abstract,
                # "pre_abstract": pre_abstract,
                "answer": item["gold"]
            }
        ) + '\n')