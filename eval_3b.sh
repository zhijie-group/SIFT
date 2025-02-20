#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

########################################
# Environment Variables & Constants
########################################

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1
export FAKETIME="@2025-01-20 15:32:02"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline
export PYTHONHASHSEED=42
export RANDOM_SEED=42

# Common Variables
VERSION='0208_llama3_2_3b_chat_AIME'
MODEL_PATH="/models/llama3_3b_instruct"
NUM_GPUS=4
OUTPUT_BASE="outputs/llama3_2_3b_AIME/gsm8k/${VERSION}"

########################################
# Stage 0: GSM8K Q2P (Query => Prediction)
########################################
sed -i "s|path='[^']*'|path='/data/gsm8k/AIME.jsonl'|" "configs/datasets/gsm8k/gsm8k_q2p.py"

python run_no_random.py \
    --datasets gsm8k_q2p \
    --hf-type chat \
    --hf-path "${MODEL_PATH}" \
    --dump-eval-details \
    --hf-num-gpus "${NUM_GPUS}" \
    -w "${OUTPUT_BASE}/abs0/"

python gsm8k_compare_token_num_v2.py \
    --base_path "${OUTPUT_BASE}/abs0/predictions/llama3_3b_instruct_hf/gsm8k.json"

########################################
# Stage 1 & 2
########################################

sed -i "s|path='[^']*'|path='/data/gsm8k/AIME.jsonl'|" "configs/datasets/gsm8k/gsm8k_q2a.py"

# Stage 1: Query => Abstract
python run_no_random.py \
    --datasets gsm8k_q2a \
    --hf-type chat \
    --hf-path "${MODEL_PATH}" \
    --dump-eval-details \
    --hf-num-gpus "${NUM_GPUS}" \
    -w "${OUTPUT_BASE}/abs1/q2a/"

python gsm8k_compare_token_num_v2.py \
    --base_path "${OUTPUT_BASE}/abs1/q2a/predictions/llama3_3b_instruct_hf/gsm8k.json"

python abs_postprocessing.py \
    --input_file "${OUTPUT_BASE}/abs1/q2a/final.jsonl" \
    --output_file "${OUTPUT_BASE}/abs1/q2a/q_a.jsonl"

# Stage 1: Query + Abstract => Prediction
sed -i "s|path='[^']*'|path='${OUTPUT_BASE}/abs1/q2a/q_a.jsonl'|" "configs/datasets/gsm8k/gsm8k_q_a2p.py"

python run_no_random.py \
    --datasets gsm8k_q_a2p \
    --hf-type chat \
    --hf-path "${MODEL_PATH}" \
    --dump-eval-details \
    --hf-num-gpus "${NUM_GPUS}" \
    -w "${OUTPUT_BASE}/abs1/q_a2p/"

python gsm8k_compare_token_num_v2.py \
    --base_path "${OUTPUT_BASE}/abs1/q_a2p/predictions/llama3_3b_instruct_hf/gsm8k.json"

# Stage 1: Abstract => Prediction
sed -i "s|path='[^']*'|path='${OUTPUT_BASE}/abs1/q2a/q_a.jsonl'|" "configs/datasets/gsm8k/gsm8k_a2p.py"

python run_no_random.py \
    --datasets gsm8k_a2p \
    --hf-type chat \
    --hf-path "${MODEL_PATH}" \
    --dump-eval-details \
    --hf-num-gpus "${NUM_GPUS}" \
    -w "${OUTPUT_BASE}/abs1/a2p/"

python gsm8k_compare_token_num_v2.py \
    --base_path "${OUTPUT_BASE}/abs1/a2p/predictions/llama3_3b_instruct_hf/gsm8k.json"

# Accumulate Stage 2 Results
python acc_stage2.py \
    --q2p "${OUTPUT_BASE}/abs0/final.jsonl" \
    --q2a "${OUTPUT_BASE}/abs1/q2a/final.jsonl" \
    --q_a2p "${OUTPUT_BASE}/abs1/q_a2p/final.jsonl" \
    --a2p "${OUTPUT_BASE}/abs1/a2p/final.jsonl" \
    --abs2improve "${OUTPUT_BASE}/abs1/abs2improve.jsonl" \
    --stage2res "${OUTPUT_BASE}/abs1/stage2res.jsonl"

########################################
# Stage 3
########################################

# Stage 3: Query + Abstract => Abstract (Forward)
sed -i "s|path='[^']*'|path='${OUTPUT_BASE}/abs1/abs2improve.jsonl'|" "configs/datasets/gsm8k/gsm8k_q_a2a.py"

python run_no_random.py \
    --datasets gsm8k_q_a2a \
    --hf-type chat \
    --hf-path "${MODEL_PATH}" \
    --dump-eval-details \
    --hf-num-gpus "${NUM_GPUS}" \
    -w "${OUTPUT_BASE}/abs2/q_a2a/"

python gsm8k_compare_token_num_v2.py \
    --base_path "${OUTPUT_BASE}/abs2/q_a2a/predictions/llama3_3b_instruct_hf/gsm8k.json"

python abs_postprocessing_v2.py \
    --input_file "${OUTPUT_BASE}/abs2/q_a2a/final.jsonl" \
    --output_file "${OUTPUT_BASE}/abs2/q_a.jsonl"

# Stage 3: Query + Abstract => Prediction
sed -i "s|path='[^']*'|path='${OUTPUT_BASE}/abs2/q_a.jsonl'|" "configs/datasets/gsm8k/gsm8k_q_a2p.py"

python run_no_random.py \
    --datasets gsm8k_q_a2p \
    --hf-type chat \
    --hf-path "${MODEL_PATH}" \
    --dump-eval-details \
    --hf-num-gpus "${NUM_GPUS}" \
    -w "${OUTPUT_BASE}/abs2/q_a2p/"

python gsm8k_compare_token_num_v2.py \
    --base_path "${OUTPUT_BASE}/abs2/q_a2p/predictions/llama3_3b_instruct_hf/gsm8k.json"

# Stage 3: Abstract => Prediction
sed -i "s|path='[^']*'|path='${OUTPUT_BASE}/abs2/q_a.jsonl'|" "configs/datasets/gsm8k/gsm8k_a2p.py"

python run_no_random.py \
    --datasets gsm8k_a2p \
    --hf-type chat \
    --hf-path "${MODEL_PATH}" \
    --dump-eval-details \
    --hf-num-gpus "${NUM_GPUS}" \
    -w "${OUTPUT_BASE}/abs2/a2p/"

python gsm8k_compare_token_num_v2.py \
    --base_path "${OUTPUT_BASE}/abs2/a2p/predictions/llama3_3b_instruct_hf/gsm8k.json"

# Accumulate Stage 3 Results
python acc_stage3.py \
    --q2p "${OUTPUT_BASE}/abs0/final.jsonl" \
    --q2a "${OUTPUT_BASE}/abs1/q2a/final.jsonl" \
    --q_a2p "${OUTPUT_BASE}/abs1/q_a2p/final.jsonl" \
    --a2p "${OUTPUT_BASE}/abs1/a2p/final.jsonl" \
    --q_a2a "${OUTPUT_BASE}/abs2/q_a2a/final.jsonl" \
    --q_a2p2 "${OUTPUT_BASE}/abs2/q_a2p/final.jsonl" \
    --a2p2 "${OUTPUT_BASE}/abs2/a2p/final.jsonl" \
    --abs2improve "${OUTPUT_BASE}/abs2/abs2improve.jsonl" \
    --stage3res "${OUTPUT_BASE}/abs2/stage3res.jsonl"

########################################
# Stage 4
########################################

# Stage 4: Prediction => Abstract (Backward)
sed -i "s|path='[^']*'|path='${OUTPUT_BASE}/abs2/abs2improve.jsonl'|" "configs/datasets/gsm8k/gsm8k_p2a.py"

python run_no_random.py \
    --datasets gsm8k_p2a \
    --hf-type chat \
    --hf-path "${MODEL_PATH}" \
    --dump-eval-details \
    --hf-num-gpus "${NUM_GPUS}" \
    -w "${OUTPUT_BASE}/abs3/p2a/"

python gsm8k_compare_token_num_v2.py \
    --base_path "${OUTPUT_BASE}/abs3/p2a/predictions/llama3_3b_instruct_hf/gsm8k.json"

python abs_postprocessing_v3.py \
    --input_file "${OUTPUT_BASE}/abs3/p2a/final.jsonl" \
    --input_file2 "${OUTPUT_BASE}/abs2/abs2improve.jsonl" \
    --output_file "${OUTPUT_BASE}/abs3/p2a/q_a.jsonl"

# Stage 4: Query + Abstract => Abstract (Forward)
sed -i "s|path='[^']*'|path='${OUTPUT_BASE}/abs3/p2a/q_a.jsonl'|" "configs/datasets/gsm8k/gsm8k_q_a2a.py"

python run_no_random.py \
    --datasets gsm8k_q_a2a \
    --hf-type chat \
    --hf-path "${MODEL_PATH}" \
    --dump-eval-details \
    --hf-num-gpus "${NUM_GPUS}" \
    -w "${OUTPUT_BASE}/abs3/q_a2a/"

python gsm8k_compare_token_num_v2.py \
    --base_path "${OUTPUT_BASE}/abs3/q_a2a/predictions/llama3_3b_instruct_hf/gsm8k.json"

python abs_postprocessing_v2.py \
    --input_file "${OUTPUT_BASE}/abs3/q_a2a/final.jsonl" \
    --output_file "${OUTPUT_BASE}/abs3/q_a.jsonl"

# Stage 4: Query + Abstract => Prediction
sed -i "s|path='[^']*'|path='${OUTPUT_BASE}/abs3/q_a.jsonl'|" "configs/datasets/gsm8k/gsm8k_q_a2p.py"

python run_no_random.py \
    --datasets gsm8k_q_a2p \
    --hf-type chat \
    --hf-path "${MODEL_PATH}" \
    --dump-eval-details \
    --hf-num-gpus "${NUM_GPUS}" \
    -w "${OUTPUT_BASE}/abs3/q_a2p/"

python gsm8k_compare_token_num_v2.py \
    --base_path "${OUTPUT_BASE}/abs3/q_a2p/predictions/llama3_3b_instruct_hf/gsm8k.json"

# Stage 4: Abstract => Prediction
sed -i "s|path='[^']*'|path='${OUTPUT_BASE}/abs3/q_a.jsonl'|" "configs/datasets/gsm8k/gsm8k_a2p.py"

python run_no_random.py \
    --datasets gsm8k_a2p \
    --hf-type chat \
    --hf-path "${MODEL_PATH}" \
    --dump-eval-details \
    --hf-num-gpus "${NUM_GPUS}" \
    -w "${OUTPUT_BASE}/abs3/a2p/"

python gsm8k_compare_token_num_v2.py \
    --base_path "${OUTPUT_BASE}/abs3/a2p/predictions/llama3_3b_instruct_hf/gsm8k.json"

# Accumulate Stage 4 Results
python acc_stage4.py \
    --q2p "${OUTPUT_BASE}/abs0/final.jsonl" \
    --q2a "${OUTPUT_BASE}/abs1/q2a/final.jsonl" \
    --q_a2p "${OUTPUT_BASE}/abs1/q_a2p/final.jsonl" \
    --a2p "${OUTPUT_BASE}/abs1/a2p/final.jsonl" \
    --q_a2a "${OUTPUT_BASE}/abs2/q_a2a/final.jsonl" \
    --q_a2p2 "${OUTPUT_BASE}/abs2/q_a2p/final.jsonl" \
    --a2p2 "${OUTPUT_BASE}/abs2/a2p/final.jsonl" \
    --p2a "${OUTPUT_BASE}/abs3/p2a/final.jsonl" \
    --q_a2a2 "${OUTPUT_BASE}/abs3/q_a2a/final.jsonl" \
    --q_a2p3 "${OUTPUT_BASE}/abs3/q_a2p/final.jsonl" \
    --a2p3 "${OUTPUT_BASE}/abs3/a2p/final.jsonl" \
    --stage4res "${OUTPUT_BASE}/abs3/stage4res.jsonl"