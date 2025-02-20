from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import NewGSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator
from opencompass.datasets import MATHEvaluator, math_postprocess_v2

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{question}\nPlease reason step by step, and put your final answer within \\boxed{}.'),
                # dict(role='HUMAN', prompt='Extract the conditions and the question from the following math problem:\n\n**Problem:**\n{question}'),
                # dict(role='HUMAN', prompt='{question}'),
                # dict(role='HUMAN', prompt='Extract the conditions and the question from the following math problem.\n---\nRequirements:\n\t1.\tConditions: Clearly list all the given information. Write each condition on a separate line, numbered sequentially.\n\t2.\tQuestion: Summarize what is being asked in one clear sentence.\n---\nOutput Format:\n```\n**Conditions:**  \n1. [Condition 1]  \n2. [Condition 2]  \n...(add more conditions as needed)  \n\n**Question:**  \n[Clearly state what is being asked.]  \n```\n---\nExample:\n\nProblem:\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\nExpected Output:\n```\n**Conditions:**  \n1. Natalia sold clips to 48 of her friends in April.  \n2. She sold half as many clips in May compared to April.  \n\n**Question:**  \nHow many clips did Natalia sell altogether in April and May?  \n```\n---\nMath Problem to Process:\n\n{question}'),
                # dict(role='HUMAN', prompt="## Task Definition\nExtract fundamental elements from math word problems using atomic decomposition methodology.\n\n## Operational Guidelines\n1. **Atomic Fact Extraction**\n   - Isolate irreducible factual units\n   - Mark source text boundaries with █[...]█\n\n2. **Core Query Formulation**\n   - Remove all known conditions\n\n---\n\n## Output Format\n```\n**Atomic Facts:**\n1. [Factual statement] █[source text]█\n2. [Next fact] █[...]█\n...(add more facts as needed)\n\n**Core Query:**\n[Condition-free fundamental question]\n```\n\n---\n\n## Example\n**Problem:**\nJames has 12 stamps, 3 more than Amy. How many stamps do they have together?\n\n**Expected Output:**\n```\n**Atomic Facts:**\n1. James possesses 12 stamps █\"James has 12 stamps\"█\n2. James's collection exceeds Amy's by 3 █\"3 more than Amy\"█\n\n**Core Query:**\nCalculate the combined total of James and Amy's stamps\n```\n\n---\n\n## Key Features\n1. **Text Anchoring** - ██ markers enable exact source mapping\n2. **Canonical Fact Pattern**\n   [Subject][Verb][Quantity][Unit]\n   Example: \"Karen bought 5 apples\" → \"Karen purchased 5 apples\"\n4. **Query Normalization**\n   - Prohibited: \"How many... (given James has 12)\"\n   - Required: \"Determine... (pure relationship)\"\n\n## Execution Parameters\n- Fact granularity: Maximum atomicity\n- Unit preservation: Maintain original measurement units\n\n---\n\n## Math Problem to Process\n{question}\n\n---\n\n## Output Format\n```\n**Atomic Facts:**\n1. [Factual statement] █[source text]█\n2. [Next fact] █[...]█\n...(add more facts as needed)\n\n**Core Query:**\n[Condition-free fundamental question]\n```\n\n---\n\nPlease provide your output following the output format without other unnecessary words."),
                # dict(role='HUMAN', prompt="Extract fundamental elements from the following query using atomic decomposition methodology.\n\n---\n\nRequirements:\n1. Conditions: Clearly list all the given information. Write each condition on a separate line, numbered sequentially.\n2. Question: Summarize what is being asked in one clear sentence. Remove all known conditions.\n\n---\n\nOutput Format:\n```\n**Conditions:**\n1. [Condition 1]\n2. [Condition 2]\n...(add more conditions as needed)  \n\n**Question:**\n[Clearly state what is being asked.]\n```\n\n---\n\nExample:\n\nQuery:\n```\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n```\n\nExpected Output:\n```\n**Conditions:**\n1. Natalia sold clips to 48 of her friends in April.\n2. She sold half as many clips in May compared to April.\n\n**Question:**\nHow many clips did Natalia sell altogether in April and May?\n```\n\n---\n\nQuery to Process:\n```\n{question}\n```\n\n---\n\nPlease provide your output strictly following the output format without other unnecessary words."),
                # dict(role='HUMAN', prompt="Extract fundamental elements from the following query using atomic decomposition methodology.\n\n---\n\nRequirements:\n1. Conditions: Clearly list all the given information. Write each condition on a separate line, numbered sequentially.\n2. Question: Summarize what is being asked in one clear sentence. Remove all known conditions.\n\n---\n\nOutput Format:\n```\n**Conditions:**\n1. [Condition 1]\n2. [Condition 2]\n...(add more conditions as needed)  \n\n**Question:**\n[Clearly state what is being asked.]\n```\n\n---\n\nExample:\n\nQuery:\n```\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n```\n\nExpected Output:\n```\n**Conditions:**\n1. Natalia sold clips to 48 of her friends in April.\n2. She sold half as many clips in May compared to April.\n\n**Question:**\nHow many clips did Natalia sell altogether in April and May?\n```\n\n---\n\nQuery to Process:\n```\n{question}\n```\n\n---\n\nPlease provide your output strictly following the output format without other unnecessary words."),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024, batch_size=8), ## max_out_len=512, batch_size=32 # 4096, 8
)

gsm8k_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'),
    pred_postprocessor=dict(type=math_postprocess_v2),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
)

gsm8k_datasets = [
    dict(
        abbr='gsm8k',
        type=NewGSM8KDataset,
        path='data/gsm8k/test_ori.jsonl',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
