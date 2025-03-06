<div align="center">
  

# <img src="docx/logo.png" width="50"> 𝕊t𝕚ck to the 𝔽ac𝕥s (SIFT)

### SIFT: Grounding LLM Reasoning in Contexts via Stickers

<!-- Authors and affiliations with improved formatting -->
Zihao Zeng, Xuyao Huang*, Boxiu Li*, Zhijie Deng<sup>†</sup><br>

Shanghai Jiao Tong University
<br>
{zengzihao, huangxuyao, lbxhaixing154, zhijied}@sjtu.edu.cn
<br>
<sup>*</sup>Equal contribution. &nbsp; <sup>†</sup>Corresponding author.
<br>

[![ArXiv](https://img.shields.io/badge/ArXiv-2502.14922-B31B1B.svg?style=flat-square)](https://arxiv.org/abs/2502.14922) &nbsp; [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-2502.14922-FFD700.svg?style=flat-square)](https://huggingface.co/papers/2502.14922)

</div>
<br>

## 🧐 What is SIFT?
**SIFT** is a novel post-training approach designed to improve the reasoning accuracy of large language models (LLMs) by mitigating context misinterpretation issues. It introduces the Sticker, a self-generated highlight that emphasizes key contextual information, ensuring more precise reasoning. Given the curated Sticker, SIFT generates two predictions---one from the original query and one from the query augmented with the Sticker. If they differ, the Sticker is sequentially refined via forward optimization (to better align the extracted facts with the query) and inverse generation (to conform with the model’s inherent tendencies) for more faithful reasoning outcomes. Experiments across diverse models (3B to 100B+) and benchmarks (e.g., GSM8K, MATH-500) show consistent performance gains. Notably, SIFT boosts DeepSeek-R1’s pass@1 accuracy on AIME2024 from 78.33% to 85.67%, setting a new state-of-the-art in open-source LLMs.

</div>

<br>

<p align="center">
<img src="docx/comparison.png" width="85%">  <!-- Slightly smaller image, adjust width as needed -->
</p>

<br>

## 🛠️ Environment Setup

### **Installation OpenCompass 🚀**

Our testing is based on OpenCompass (version 0.3.4). For installation and usage instructions, refer to [OpenCompass GitHub](https://github.com/open-compass/opencompass).

### **Dataset 📂**
Create a new file named `test_ori.jsonl` in the directory `/opencompass/data/gsm8k/`, and ensure that the format of this dataset matches that of `/opencompass/data/gsm8k/test.jsonl`. You can replace the content with the dataset you want to test.

We also provide some datasets for evaluation in the `data` folder. You may need to modify the key names in the jsonl file, such as changing `problem` to `question`.They can be used to replace the content in `/opencompass/data/gsm8k/test_ori.jsonl`.


### **Preparation Steps(using GSM8K as an example) 🔧**

#### **File Replacements 🔄**
Replace the following files with the provided ones:
- Replace `/opencompass/models/huggingface_above_v4_33.py` with `huggingface_above_v4_33.py`.
- Replace `/opencompass/openicl/icl_inferencer/icl_gen_inferencer.py` with `icl_gen_inferencer.py`.
- Replace `/opencompass/openicl/icl_inferencer/icl_base_inferencer.py` with `icl_base_inferencer.py`.

#### **Add Dataset Configurations 📑**
Copy the following files into `/opencompass/configs/datasets/gsm8k`:
-  `gsm8k_a2p.py`
-  `gsm8k_p2a.py`
-  `gsm8k_q_a2a.py`
-  `gsm8k_q_a2p.py`
-  `gsm8k_q2a.py`
-  `gsm8k_q2p.py`

#### **Add Additional Files 📂**
Copy the following files into `/opencompass`:
-  `gsm8k_compare_token_num_v2.py`
-  `acc_stage2.py`
-  `acc_stage3.py`
-  `acc_stage4.py`
-  `abs_postprocessing.py`
-  `abs_postprocessing_v2.py`
-  `abs_postprocessing_v3.py`
-  `eval_3b.sh`
-  `run_no_random.py`

Copy the following file into `/opencompass/opencompass/datasets`:
-  `gsm8k.py`


✅ These steps ensure the correct setup of the environment and necessary configurations for Opencompass.

✅ If you want to test a different dataset, such as math-500, simply replace the content of `/opencompass/data/gsm8k/test_ori.jsonl` with the math-500 dataset, keeping the format unchanged.

## 🚀 Running the Experiments
After setting up the environment, run the experiment with:
```
bash eval_3b.sh
```



## Acknowledgements
Our work mainly builds upon [OpenCompass](https://github.com/open-compass/opencompass). We also used the open-source models [Qwen](https://github.com/QwenLM/Qwen2.5) and [LLaMA](https://github.com/meta-llama/llama3) for local evaluation. The DeepSeek-R1 model was evaluated via API calls.The evaluation results of the OpenAI o-series models for AIME24 and AIME25 are derived from [AIME-Preview](https://github.com/GAIR-NLP/AIME-Preview).


## Citation
If you find our paper or codebase useful, please consider citing:
```
@misc{zeng2025siftgroundingllmreasoning,
      title={SIFT: Grounding LLM Reasoning in Contexts via Stickers}, 
      author={Zihao Zeng and Xuyao Huang and Boxiu Li and Zhijie Deng},
      year={2025},
      eprint={2502.14922},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14922}, 
}
```
