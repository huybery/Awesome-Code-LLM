<div align="center">
  <h1>👨‍💻 Awesome Code LLM</h1>
  <a href="https://awesome.re">
    <img src="https://awesome.re/badge.svg" alt="Awesome">
  </a>
  <a href="https://img.shields.io/badge/PRs-Welcome-red">
    <img src="https://img.shields.io/badge/PRs-Welcome-red" alt="PRs Welcome">
  </a>
  <a href="https://img.shields.io/github/last-commit/huybery/Awesome-Code-LLM?color=green">
    <img src="https://img.shields.io/github/last-commit/huybery/Awesome-Code-LLM?color=green" alt="Last Commit">
  </a>
</div>

![](code-banner.png)

## 🧵 Table of Contents

- [🧵 Table of Contents](#-table-of-contents)
- [🚀 Leaderboard](#-leaderboard)
- [💡 Evaluation Toolkit:](#-evaluation-toolkit)
- [📚 Paper](#-paper)
  - [▶️ Pre-Training](#️-pre-training)
  - [▶️ Instruction Tuning](#️-instruction-tuning)
  - [▶️ Alignment with Feedback](#️-alignment-with-feedback)
  - [▶️ Prompting](#️-prompting)
  - [▶️ Evaluation \& Benchmark](#️-evaluation--benchmark)
  - [▶️ Using LLMs while coding](#️-using-llms-while-coding)
- [🙌 Contributors](#-contributors)
- [Cite as](#cite-as)
- [Acknowledgement](#acknowledgement)
- [Star History](#star-history)

## 🚀 Leaderboard

<p align="center"> <b>Central Leaderboard</b> (Sort by HumanEval Pass@1) </p>

| Model                    | Params | HumanEval | MBPP | HF                                                            | Source                                                  |
| ------------------------ | ------ | --------- | ---- | ------------------------------------------------------------- | ------------------------------------------------------- |
| o1-mini-2024-09-12          | -      | 97.6      | 93.9 | - | [paper](https://arxiv.org/abs/2409.12186) |
| o1-preview-2024-09-12          | -      | 95.1      | 93.4 | - | [paper](https://arxiv.org/abs/2409.12186) |
| Qwen2.5-Coder-32B-Instruct      | 32B     | 92.7      | 90.2 | [ckpt](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)       | [github](https://github.com/QwenLM/Qwen2.5-Coder)         |
| Claude-3.5-Sonnet-20241022          | -      | 92.1      | 91.0 | - | [paper](https://arxiv.org/abs/2409.12186) |
| GPT-4o-2024-08-06          | -      | 92.1      | 86.8 | - | [paper](https://arxiv.org/abs/2409.12186) |
| Qwen2.5-Coder-14B-Instruct      | 14B     | 89.6      | 86.2 | [ckpt](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct)       | [github](https://github.com/QwenLM/Qwen2.5-Coder)         |
| Claude-3.5-Sonnet-20240620          | -      | 89.0      | 87.6 | - | [paper](https://arxiv.org/abs/2409.12186) |
| GPT-4o-mini-2024-07-18          | -      | 87.8      | 86.0 | - | [paper](https://arxiv.org/abs/2409.12186) |
| Qwen2.5-Coder-7B-Instruct      | 7B     | 88.4      | 83.5 | [ckpt](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)       | [github](https://github.com/QwenLM/Qwen2.5-Coder)         |
| DS-Coder-V2-Instruct      | 21/236B     | 85.4      | 89.4 | [ckpt](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct)       | [github](https://github.com/deepseek-ai/DeepSeek-Coder-V2)         |
| Qwen2.5-Coder-3B-Instruct      | 3B     | 84.1      | 73.6 | [ckpt](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct)       | [github](https://github.com/QwenLM/Qwen2.5-Coder)         |
| DS-Coder-V2-Lite-Instruct      | 2.4/16B     | 81.1      | 82.8 | [ckpt](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct)       | [github](https://github.com/deepseek-ai/DeepSeek-Coder-V2)         |
| CodeQwen1.5-7B-Chat      | 7B     | 83.5      | 70.6 | [ckpt](https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat)       | [github](https://github.com/QwenLM/CodeQwen1.5)         |
| DeepSeek-Coder-Instruct  | 33B    | 79.3      | 70.0 | [ckpt](https://hf.co/deepseek-ai/deepseek-coder-33b-instruct) | [github](https://github.com/deepseek-ai/DeepSeek-Coder) |
| DeepSeek-Coder-Instruct  | 7B     | 78.6      | 65.4 | [ckpt](https://hf.co/deepseek-ai/deepseek-coder-33b-instruct) | [github](https://github.com/deepseek-ai/DeepSeek-Coder) |
| GPT-3.5-Turbo (latest)   | -      | 76.2      | 70.8 |                                                               | [github](https://github.com/deepseek-ai/DeepSeek-Coder) |
| Qwen2.5-Coder-1.5B-Instruct      | 1.5B     | 70.7      | 69.2 | [ckpt](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)       | [github](https://github.com/QwenLM/Qwen2.5-Coder)         |
| Code-Llama               | 34B    | 62.2      | 61.2 |                                                               | [paper](https://arxiv.org/abs/2308.12950)               |
| Qwen2.5-Coder-0.5B-Instruct      | 0.5B     | 61.6      | 52.4 | [ckpt](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)       | [github](https://github.com/QwenLM/Qwen2.5-Coder)         |
| Pangu-Coder2             | 15B    | 61.6      |      |                                                               | [paper](https://arxiv.org/abs/2307.14936)               |
| WizardCoder-15B          | 15B    | 57.3      | 51.8 | [ckpt](https://hf.co/WizardLM/WizardCoder-15B-V1.0)           | [paper](https://arxiv.org/abs/2306.08568)               |
| CodeQwen1.5-7B      | 7B     | 51.8      | 61.8 | [ckpt](https://huggingface.co/Qwen/CodeQwen1.5-7B)            | [github](https://github.com/QwenLM/CodeQwen1.5)         |
| Code-Davinci-002         | -      | 47.0      |      |                                                               | [paper](https://arxiv.org/abs/2107.03374)               |
| StarCoder-15B (Prompted) | 15B    | 40.8      | 49.5 | [ckpt](https://hf.co/bigcode/starcoder)                       | [paper](https://arxiv.org/abs/2305.06161)               |
| PaLM 2-S                 | -      | 37.6      | 50.0 |                                                               | [paper](https://arxiv.org/abs/2204.02311)               |
| PaLM-Coder-540B          | 540B   | 36.0      | 47.0 |                                                               | [paper](https://arxiv.org/abs/2204.02311)               |
| InstructCodeT5+          | 16B    | 35.0      |      |                                                               | [paper](https://arxiv.org/abs/2305.07922)               |
| StarCoder-15B            | 15B    | 33.6      | 52.7 | [ckpt](https://hf.co/bigcode/starcoder)                       | [paper](https://arxiv.org/abs/2305.06161)               |
| Code-Cushman-001         | -      | 33.5      | 45.9 |                                                               | [paper](https://arxiv.org/abs/2107.03374)               |
| CodeT5+                  | 16B    | 30.9      |      |                                                               | [paper](https://arxiv.org/abs/2305.07922)               |
| LLaMA2-70B               | 70B    | 29.9      |      | [ckpt](https://hf.co/meta-llama/Llama-2-70b-hf)               | [paper](https://arxiv.org/abs/2307.09288)               |
| CodeGen-16B-Mono         | 16B    | 29.3      | 35.3 |                                                               | [paper](https://arxiv.org/abs/2203.13474)               |
| PaLM-540B                | 540B   | 26.2      | 36.8 |                                                               | [paper](https://arxiv.org/abs/2204.02311)               |
| LLaMA-65B                | 65B    | 23.7      | 37.7 |                                                               | [paper](https://arxiv.org/abs/2302.13971)               |
| CodeGeeX                 | 13B    | 22.9      | 24.4 |                                                               | [paper](https://arxiv.org/abs/2303.17568)               |
| LLaMA-33B                | 33B    | 21.7      | 30.2 |                                                               | [paper](https://arxiv.org/abs/2302.13971)               |
| CodeGen-16B-Multi        | 16B    | 18.3      | 20.9 |                                                               | [paper](https://arxiv.org/abs/2203.13474)               |
| AlphaCode                | 1.1B   | 17.1      |      |                                                               | [paper](https://arxiv.org/abs/2203.07814)               |

| Leaderboard                          | Access                                                                            |
| :----------------------------------: | ----------------------------------------------------------------------------------|
| Big Code Models Leaderboard          | [[Source](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)]      |
| BIRD                                 | [[Source](https://bird-bench.github.io)]                                          |
| CanAiCode Leaderboard                | [[Source](https://huggingface.co/spaces/mike-ravkine/can-ai-code-results)]        |
| Coding LLMs Leaderboard              | [[Source](https://leaderboard.tabbyml.com)]                                       |
| CRUXEval Leaderboard                 | [[Source](https://crux-eval.github.io/leaderboard.html)]                          |
| EvalPlus                             | [[Source](https://evalplus.github.io/leaderboard.html)]                           |
| HumanEval.jl                         | [[Source](https://github.com/01-ai/HumanEval.jl)]                                 |
| InfiCoder-Eval                       | [[Source](https://infi-coder.github.io/inficoder-eval)]                           |
| InterCode                            | [[Source](https://intercode-benchmark.github.io)]                                 |
| Program Synthesis Models Leaderboard | [[Source](https://accubits.com/open-source-program-synthesis-models-leaderboard)] |
| Spider                               | [[Source](https://yale-lily.github.io/spider)]                                    |

## 💡 Evaluation Toolkit:

- [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness): A framework for the evaluation of autoregressive code generation language models.
- [code-eval](https://github.com/abacaj/code-eval): A framework for the evaluation of autoregressive code generation language models on HumanEval.

## 📚 Paper

### ▶️ Pre-Training

1. **Evaluating Large Language Models Trained on Code** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2107.03374)] *Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto. et al.* 2021.07

2. **CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis** `ICLR23`
  
    [[Paper](https://arxiv.org/abs/2203.13474)] *Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, Caiming Xiong.* 2022.03

3. **ERNIE-Code: Beyond English-Centric Cross-lingual Pretraining for Programming Languages** `ACL23 (Findings)`

    [[Paper](https://aclanthology.org/2023.findings-acl.676.pdf)][[Repo](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-code)] *Yekun Chai, Shuohuan Wang, Chao Pang, Yu Sun, Hao Tian, and Hua Wu.* 2022.12

4. **SantaCoder: don't reach for the stars!** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2301.03988)] *Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff. et al.* 2023.01

5. **CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2303.17568)] *Qinkai Zheng, Xiao Xia, Xu Zou, Yuxiao Dong, Shan Wang, Yufei Xue, Zihan Wang, Lei Shen, Andi Wang, Yang Li, Teng Su, Zhilin Yang, Jie Tang.* 2023.03

6. **CodeGen2: Lessons for Training LLMs on Programming and Natural Languages** `ICLR23`
  
    [[Paper](https://arxiv.org/abs/2305.02309)] *Erik Nijkamp, Hiroaki Hayashi, Caiming Xiong, Silvio Savarese, Yingbo Zhou.* 2023.05

7. **StarCoder: may the source be with you!** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2305.06161)] *Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou. et al.* 2023.05

8. **CodeT5+: Open Code Large Language Models for Code Understanding and Generation** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2305.07922)] *Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi D.Q. Bui, Junnan Li, Steven C.H. Hoi.* 2023.05

9. **Textbooks Are All You Need** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2306.11644)] *Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi. et al.* 2023.06

10. **Code Llama: Open Foundation Models for Code** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2308.12950)] *Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat. et al.* 2023.08

11. **DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence** `Preprint`

    [[Paper](https://arxiv.org/abs/2401.14196)] *Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen et al.* 2024.01

12. **StarCoder 2 and The Stack v2: The Next Generation** `Preprint`

    [[Paper](https://arxiv.org/abs/2402.19173)] *Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang et al.* 2024.02

13. **DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence** `Preprint`

    [[Paper](https://arxiv.org/abs/2406.11931)] *DeepSeek-AI, Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu, Y. Wu et al.* 2024.06

14. **Qwen2.5-Coder Technical Report** `Preprint`

    [[Paper](https://arxiv.org/abs/2409.12186)] *Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang et al.* 2024.09

15. **OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models** `Preprint`

    [[Paper](https://arxiv.org/abs/2411.04905)] *Siming Huang, Tianhao Cheng, J.K. Liu, Jiaran Hao, Liuyihan Song, Yang Xu, J. Yang, J.H. Liu et al.* 2024.11

### ▶️ Instruction Tuning

1. **Code Alpaca: An Instruction-following LLaMA Model trained on code generation instructions**

    [[Repo](https://github.com/sahil280114/codealpaca)] *Sahil Chaudhary.* 2023

2. **WizardCoder: Empowering Code Large Language Models with Evol-Instruct** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2306.08568)] *Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, Daxin Jiang.* 2023.07

3. **OctoPack: Instruction Tuning Code Large Language Models** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2308.07124)][[Repo](https://github.com/bigcode-project/octopack)] *Niklas Muennighoff, Qian Liu, Armel Zebaze, Qinkai Zheng, Binyuan Hui, Terry Yue Zhuo, Swayam Singh, Xiangru Tang, Leandro von Werra, Shayne Longpre.* 2023.08

4. **Magicoder: Source Code Is All You Need** `Preprint`

    [[Paper](https://arxiv.org/abs/2312.02120)][[Repo](https://github.com/ise-uiuc/magicoder)] *Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, Lingming Zhang* 2023.12


### ▶️ Alignment with Feedback

1. **CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning** `NeurIPS22`
  
    [[Paper](https://arxiv.org/abs/2207.01780)] *Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, Steven C.H. Hoi.* 2022.07 

2. **Execution-based Code Generation using Deep Reinforcement Learning** `TMLR23`
  
    [[Paper](https://arxiv.org/abs/2301.13816)] *Parshin Shojaee, Aneesh Jain, Sindhu Tipirneni, Chandan K. Reddy.* 2023.01 

3. **RLTF: Reinforcement Learning from Unit Test Feedback** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2307.04349)] *Jiate Liu, Yiqin Zhu, Kaiwen Xiao, Qiang Fu, Xiao Han, Wei Yang, Deheng Ye.* 2023.07 

4. **PanGu-Coder2: Boosting Large Language Models for Code with Ranking Feedback** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2307.14936)] *Bo Shen, Jiaxin Zhang, Taihong Chen, Daoguang Zan, Bing Geng, An Fu, Muhan Zeng, Ailun Yu, Jichuan Ji, Jingyang Zhao, Yuenan Guo, Qianxiang Wang.* 2023.07 

5. **PLUM: Preference Learning Plus Test Cases Yields Better Code Language Models** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2406.06887)] *Dylan Zhang, Shizhe Diao, Xueyan Zou, Hao Peng.* 2024.06

### ▶️ Prompting

1. **CodeT: Code Generation with Generated Tests** `ICLR23`
  
    [[Paper](https://arxiv.org/abs/2207.10397)] *Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, Weizhu Chen.* 2022.07

2. **Coder Reviewer Reranking for Code Generation** `ICML23`
  
    [[Paper](https://arxiv.org/abs/2211.16490)] *Tianyi Zhang, Tao Yu, Tatsunori B Hashimoto, Mike Lewis, Wen-tau Yih, Daniel Fried, Sida I Wang.* 2022.11

3. **LEVER: Learning to Verify Language-to-Code Generation with Execution** `ICML23`
  
    [[Paper](https://arxiv.org/abs/2302.08468)] *Ansong Ni, Srini Iyer, Dragomir Radev, Ves Stoyanov, Wen-tau Yih, Sida I. Wang, Xi Victoria Lin.* 2023.02

4. **Teaching Large Language Models to Self-Debug** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2304.05128)] *Xinyun Chen, Maxwell Lin, Nathanael Schärli, Denny Zhou.* 2023.06

5. **Demystifying GPT Self-Repair for Code Generation** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2306.09896)] *Theo X. Olausson, Jeevana Priya Inala, Chenglong Wang, Jianfeng Gao, Armando Solar-Lezama.* 2023.06

6. **SelfEvolve: A Code Evolution Framework via Large Language Models** `Preprint`
   
    [[Paper](https://arxiv.org/abs/2306.02907)] *Shuyang Jiang, Yuhao Wang, Yu Wang.* 2023.06

7. **Debug like a Human: A Large Language Model Debugger via Verifying Runtime Execution Step-by-step** `ACL24`

    [[Paper](https://arxiv.org/abs/2402.16906)] *Li Zhong, Zilong Wang, Jingbo Shang.* 2024.02

8. **From Code to Correctness: Closing the Last Mile of Code Generation with Hierarchical Debugging.** `Preprint`

    [[Paper](https://arxiv.org/abs/2410.01215)][[Repo](https://github.com/YerbaPage/MGDebugger)] *Yuling Shi, Songsong Wang, Chengcheng Wan, Xiaodong Gu.* 2024.10


### ▶️ Evaluation & Benchmark

1. **Measuring Coding Challenge Competence With APPS** `NeurIPS21`

    > Named APPS
  
    [[Paper](https://arxiv.org/abs/2108.07732)][[Repo](https://github.com/hendrycks/apps)] *Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, Jacob Steinhardt.* 2021.05 

2. **Program Synthesis with Large Language Models** `Preprint`

    > Named MBPP
  
    [[Paper](https://arxiv.org/abs/2108.07732)] *Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, Charles Sutton.* 2021.08 

3. **DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation** `ICML23`

    [[Paper](https://arxiv.org/abs/2211.11501)] *Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, Luke Zettlemoyer, Scott Wen-tau Yih, Daniel Fried, Sida Wang, Tao Yu.* 2022.11 

4. **RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems** `Preprint`

    [[Paper](https://arxiv.org/abs/2306.03091)] *Tianyang Liu, Canwen Xu, Julian McAuley.* 2023.06 

5. **Can ChatGPT replace StackOverflow? A Study on Robustness and Reliability of Large Language Model Code Generation** `Preprint`

    [[Paper](https://arxiv.org/abs/2308.10335)] *Li Zhong, Zilong Wang.* 2023.08

6. **RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation** `EMNLP23`

    [[Paper](https://arxiv.org/abs/2303.12570)] *Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, Weizhu Chen.* 2023.10

7. **CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion** `Neurips23`

    [[Paper](https://arxiv.org/abs/2310.11248)] *Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Hantian Ding, Ming Tan, Nihal Jain, Murali Krishna Ramanathan. et al.* 2023.11

8. **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?** `ICLR24`

    [[Paper](https://arxiv.org/abs/2310.06770)] *YCarlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan.* 2023.10

9. **DevBench: A Comprehensive Benchmark for Software Development** `Preprint`

    [[Paper](https://arxiv.org/abs/2403.08604)][[Repo](https://github.com/open-compass/DevBench)] *Bowen Li, Wenhan Wu, Ziwei Tang, Lin Shi, John Yang, Jinyang Li, Shunyu Yao, Chen Qian, Binyuan Hui, Qicheng Zhang, Zhiyin Yu, He Du, Ping Yang, Dahua Lin, Chao Peng, Kai Chen* 2024.3

10. **LongCoder: A Long-Range Pre-trained Language Model for Code Completion** `ICML23`

    [[Paper](https://arxiv.org/abs/2306.14893)] *Daya Guo, Canwen Xu, Nan Duan, Jian Yin, Julian McAuley.* 2023.10

11. **Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing** `Preprint`

    [[Paper](https://arxiv.org/abs/2305.18584)] *Jiayi Wei, Greg Durrett, Isil Dillig.* 2023.5

12. **Automating Code Review Activities by Large-Scale Pre-training** `Preprint`

    [[Paper](https://arxiv.org/abs/2203.09095)] *JZhiyu Li, Shuai Lu, Daya Guo, Nan Duan, Shailesh Jannu, Grant Jenks, Deep Majumder, Jared Green, Alexey Svyatkovskiy, Shengyu Fu, Neel Sundaresan.* 2022.10


13. **BioCoder: A Benchmark for Bioinformatics Code Generation with Large Language Models** `ISMB 2024`

    [[Paper](https://arxiv.org/abs/2308.16458)] *Xiangru Tang, Bill Qian, Rick Gao, Jiakang Chen, Xinyun Chen, Mark Gerstein.* 2023.08




### ▶️ Using LLMs while coding

1.  **Awesome-DevAI: A list of resources about using LLMs while building software** `Awesome`

    [[Repo](https://github.com/continuedev/Awesome-DevAI)] *Ty Dunn, Nate Sesti.* 2023.10

## 🙌 Contributors

<a href="https://github.com/huybery"><img src="https://avatars.githubusercontent.com/u/13436140?v=4"  width="50" /></a>
<a href="https://github.com/Yangjiaxi"><img src="https://avatars.githubusercontent.com/u/6203054?v=4"  width="50" /></a>
<a href="https://github.com/GanjinZero"><img src="https://avatars.githubusercontent.com/u/19466330?v=4"  width="50" /></a>
<a href="https://github.com/TyDunn"><img src="https://avatars.githubusercontent.com/u/13314504?v=4"  width="50" /></a>
<a href="https://github.com/Hambaobao"><img src="https://avatars.githubusercontent.com/u/48345096?v=4"  width="50" /></a>

This is an active repository and your contributions are always welcome! If you have any question about this opinionated list, do not hesitate to contact me `huybery@gmail.com`.

## Cite as

```
@software{awesome-code-llm,
  author = {Binyuan Hui},
  title = {An awesome and curated list of best code-LLM for research},
  howpublished = {\url{https://github.com/huybery/Awesome-Code-LLM}},
  year = 2023,
}
```

## Acknowledgement

This project is inspired by [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=huybery/Awesome-Code-LLM&type=Date)](https://star-history.com/#huybery/Awesome-Code-LLM&Date)


**[⬆ Back to ToC](#table-of-contents)**