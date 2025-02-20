from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import NewGSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator
from opencompass.datasets import MATHEvaluator, math_postprocess_v2

gsm8k_reader_cfg = dict(input_columns=['question','prediction'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                # dict(role='HUMAN', prompt="Given the prediction provided below, reverse-engineer the abstract that led to it. The abstract should include both the conditions and the question.\n\n---\n\nAbstract Format:\n```\n**Conditions:**\n1. [Condition 1]\n2. [Condition 2]\n...(add more conditions as needed)  \n\n**Question:**\n[Clearly state what is being asked.]\n```\n\n---\n\nRequirements:\n1. Conditions: \n    - Clearly list all the given information. \n    - Write each condition on a separate line, numbered sequentially. \n    - EACH CONDITION MUST BE ATOMIC AND INDIVISIBLE (i.e., it cannot be divided into two sub-conditions). \n    - DO NOT INCLUDE ANY PART OF THE REASONING PROCESS!!!\n2. Question: \n    - Summarize what is being asked in one clear sentence. \n    - Remove all known conditions.\n\n---\n\nExample:\n\nPrediction:\n```\nTo find the total number of clips Natalia sold, we need to calculate the number of clips she sold in April and May separately, and then add them together.\n\nIn April, Natalia sold 48 clips.\n\nIn May, she sold half as many clips as in April. To find half of 48, we can divide 48 by 2:\n\n48 ÷ 2 = 24\n\nSo, in May, Natalia sold 24 clips.\n\nNow, let's add the number of clips she sold in April and May:\n\n48 (April) + 24 (May) = 72\n\nTherefore, Natalia sold 72 clips altogether in April and May.\n\n\\boxed{72}\n```\n\nExpected Output:\n```\n**Conditions:**\n1. Natalia sold clips to 48 of her friends in April.\n2. She sold half as many clips in May compared to April.\n\n**Question:**\nHow many clips did Natalia sell altogether in April and May?\n```\n\n---\n\nPrediction to Process:\n```\n{prediction}\n```\n\n---\n\nPlease provide your output strictly following the ABSTRACT FORMAT without other unnecessary words."),
                # dict(role='HUMAN', prompt="Given the prediction provided below, reverse-engineer the abstract that led to it. The abstract should include both the conditions and the question.\n\n---\n\nAbstract Format:\n```\n**Conditions:**\n1. [Condition 1]\n2. [Condition 2]\n...(add more conditions as needed)  \n\n**Question:**\n[Clearly state what is being asked.]\n```\n\n---\n\nRequirements:\n1. Conditions: \n    - Clearly list all the given information. \n    - Write each condition on a separate line, numbered sequentially. \n    - EACH CONDITION MUST BE ATOMIC AND INDIVISIBLE (i.e., it cannot be divided into two sub-conditions). \n    - DO NOT INCLUDE ANY PART OF THE REASONING PROCESS!!!\n2. Question: \n    - Summarize what is being asked in one clear sentence. \n    - Remove all known conditions.\n\n---\n\nExample:\n\nPrediction:\n```\nTo find the total number of clips Natalia sold, we need to calculate the number of clips she sold in April and May separately, and then add them together.\n\nIn April, Natalia sold 48 clips.\n\nIn May, she sold half as many clips as in April. To find half of 48, we can divide 48 by 2:\n\n48 ÷ 2 = 24\n\nSo, in May, Natalia sold 24 clips.\n\nNow, let's add the number of clips she sold in April and May:\n\n48 (April) + 24 (May) = 72\n\nTherefore, Natalia sold 72 clips altogether in April and May.\n\n\\boxed72\n```\n\nExpected Output:\n```\n**Conditions:**\n1. Natalia sold clips to 48 of her friends in April.\n2. She sold half as many clips in May compared to April.\n\n**Question:**\nHow many clips did Natalia sell altogether in April and May?\n```\n\n---\n\nPrediction to Process:\n```\n{prediction}\n```\n\n---\n\nPlease provide your output strictly following the ABSTRACT FORMAT without other unnecessary words."),
                dict(role='HUMAN', prompt="Given the prediction provided below, reverse-engineer the abstract that led to it. The abstract should include both the conditions and the question.\n\n---\n\nAbstract Format:\n```\n**Conditions:**\n1. [Condition 1]\n2. [Condition 2]\n...(add more conditions as needed)  \n\n**Question:**\n[Clearly state what is being asked.]\n```\n\n---\n\nRequirements:\n1. Conditions: \n    - Clearly list all the given information. \n    - Write each condition on a separate line, numbered sequentially. \n    - EACH CONDITION MUST BE ATOMIC AND INDIVISIBLE (i.e., it cannot be divided into two sub-conditions). \n    - DO NOT INCLUDE ANY PART OF THE REASONING PROCESS!!!\n2. Question: \n    - Summarize what is being asked in one clear sentence. \n    - Remove all known conditions.\n\n---\n\nExample:\n\nPrediction:\n```\nTo find the total number of clips Natalia sold, we need to calculate the number of clips she sold in April and May separately, and then add them together.\n\nIn April, Natalia sold 48 clips.\n\nIn May, she sold half as many clips as in April. To find half of 48, we can divide 48 by 2:\n\n48 ÷ 2 = 24\n\nSo, in May, Natalia sold 24 clips.\n\nNow, let's add the number of clips she sold in April and May:\n\n48 (April) + 24 (May) = 72\n\nTherefore, Natalia sold 72 clips altogether in April and May.\n\n\\boxed72\n```\n\nExpected Output:\n```\n**Conditions:**\n1. Natalia sold clips to 48 of her friends in April.\n2. She sold half as many clips in May compared to April.\n\n**Question:**\nHow many clips did Natalia sell altogether in April and May?\n```\n\n---\n\nPrediction to Process:\n```\n{prediction}\n```\n\n---\n\nPlease provide your output strictly following the ABSTRACT FORMAT without other unnecessary words."),
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
        path='outputs/llama3_2_3b_chat/gsm8k/0215_1_dim_sift/epoch16/abs2/abs2improve.jsonl',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
