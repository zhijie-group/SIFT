import json
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, help='Path to the input JSONL file')
parser.add_argument('--input_file2', type=str, help='Path to the input JSONL file')
parser.add_argument('--output_file', type=str, help='Path to the output JSONL file')

input_file = parser.parse_args().input_file
input_file2 = parser.parse_args().input_file2
output_file = parser.parse_args().output_file

with open(input_file, 'r') as f:
    data = [json.loads(line) for line in f]

with open(input_file2, 'r') as f:
    data2 = [json.loads(line) for line in f]

with open(output_file, 'w') as f:
    for item, item2 in zip(data, data2):
        abstract = item['prediction']
        if "**Conditions:**" in abstract:
            abstract = "**Conditions:**" + abstract.split("**Conditions:**")[1]
            if "```" in abstract:
                abstract = abstract.split("```")[0].strip()
        # pre_abstract = item2['abstract']
        # if abstract == pre_abstract:
        #     pass
        # else:
        f.write(json.dumps(
            {
                "question": item2['question'],
                "abstract": abstract,
                "answer": item["gold"]
            }
        ) + '\n')