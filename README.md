<div align="center">

# SIFT

### SIFT: Grounding LLM Reasoning in Contexts via Stickers

<!-- Authors and affiliations with improved formatting -->
Zihao Zeng, Xuyao Huang*, Boxiu Li*, Zhijie Deng<sup>†</sup><br>

Shanghai Jiao Tong University
<br>
{zengzihao, huangxuyao, lbxhaixing154, zhijied}@sjtu.edu.cn
<br>
<sup>*</sup>Equal contribution. &nbsp; <sup>†</sup>Corresponding author.
<br>

</div>

<br>

## What is SIFT?
**SIFT** is a novel post-training approach designed to improve the reasoning accuracy of large language models (LLMs) by mitigating context misinterpretation issues. It introduces the Sticker, a self-generated highlight that emphasizes key contextual information, ensuring more precise reasoning. SIFT refines predictions by comparing responses with and without the Sticker, using forward optimization and inverse generation to align facts with queries. Experiments across diverse models (3B to 100B+) and benchmarks (e.g., GSM8K, MATH-500) show consistent performance gains. Notably, SIFT boosts DeepSeek-R1’s pass@1 accuracy on AIME2024 from 78.33% to 85.67%, setting a new state-of-the-art in open-source LLMs.

</div>

<br>

## 🛠️ Environment Setup

# 📌 Installation Opencompass

### Create your virtual environment

```sh
conda create --name opencompass python=3.10 -y
conda activate opencompass
pip install -U opencompass






