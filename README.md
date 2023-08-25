# 👨‍💻 Awesome-Code-LLM [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![PR](https://img.shields.io/badge/PRs-Welcome-red)](https://img.shields.io/badge/PRs-Welcome-red) [![COMMIT](https://img.shields.io/github/last-commit/huybery/Awesome-Code-LLM?color=green)](https://img.shields.io/github/last-commit/huybery/Awesome-Code-LLM?color=green)

![](code-banner.png)

## 🧵 Table of Contents

- [👨‍💻 Awesome-Code-LLM   ](#-awesome-code-llm---)
  - [🧵 Table of Contents](#-table-of-contents)
  - [🚀 Leaderboard](#-leaderboard)
  - [📚 Paper](#-paper)
    - [▶️ Pre-Training](#️-pre-training)
    - [▶️ Instruction Tuning](#️-instruction-tuning)
    - [▶️ Alignment with Feedback](#️-alignment-with-feedback)
    - [▶️ Prompting](#️-prompting)
    - [▶️ Evaluation \& Benchmark](#️-evaluation--benchmark)
  - [🙌 Contributors](#-contributors)
  - [Cite as](#cite-as)
  - [Acknowledgement](#acknowledgement)
  - [Star History](#star-history)

## 🚀 Leaderboard

<p align="center"> <b>Leaderboard</b> (Sort by HumanEval Pass@1) </p>

| Rank | Model                    | Params | HumanEval | MBPP | HF                                                  | Paper                                     |
| ---- | ------------------------ | ------ | --------- | ---- | --------------------------------------------------- | ----------------------------------------- |
| 1    | GPT-4 + Reflexion        | ?      | 91.0      | 77.1 |                                                     | [paper](https://arxiv.org/abs/2303.11366) |
| 2    | GPT-4                    | ?      | 67.0      |      |                                                     | [paper](https://arxiv.org/abs/2303.08774) |
| 3    | Code-Llama               | 34B    | 62.2      | 61.2 |                                                     | [paper](https://scontent-hkg4-1.xx.fbcdn.net/v/t39.2365-6/369856151_1754812304950972_1159666448927483931_n.pdf?_nc_cat=107&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=BnkB4kcpz5AAX9kJmnq&_nc_ht=scontent-hkg4-1.xx&oh=00_AfDk1DUUkNXLvMWb8tjPbZ_d1mvrtdl0AkTkhwf7svqHPw&oe=64ECB20F) |
| 4    | Pangu-Coder2             | 15B    | 61.6      |      |                                                     | [paper](https://arxiv.org/abs/2307.14936) |
| 5    | WizardCoder-15B          | 15B    | 57.3      | 51.8 | [ckpt](https://hf.co/WizardLM/WizardCoder-15B-V1.0) | [paper](https://arxiv.org/abs/2306.08568) |
| 6    | GPT-3.5                  | ?      | 48.1      |      |                                                     | [paper](https://arxiv.org/abs/2303.08774) |
| 7    | Code-Davinci-002         | ?      | 47.0      |      |                                                     | [paper](https://arxiv.org/abs/2107.03374) |
| 8    | StarCoder-15B (Prompted) | 15B    | 40.8      | 49.5 | [ckpt](https://hf.co/bigcode/starcoder)             | [paper](https://arxiv.org/abs/2305.06161) |
| 9    | PaLM 2-S                 | ?      | 37.6      | 50.0 |                                                     | [paper](https://arxiv.org/abs/2204.02311) |
| 10   | PaLM-Coder-540B          | 540B   | 36.0      | 47.0 |                                                     | [paper](https://arxiv.org/abs/2204.02311) |
| 11   | InstructCodeT5+          | 16B    | 35.0      |      |                                                     | [paper](https://arxiv.org/abs/2305.07922) |
| 12   | StarCoder-15B            | 15B    | 33.6      | 52.7 | [ckpt](https://hf.co/bigcode/starcoder)             | [paper](https://arxiv.org/abs/2305.06161) |
| 13   | Code-Cushman-001         | ?      | 33.5      | 45.9 |                                                     | [paper](https://arxiv.org/abs/2107.03374) |
| 14   | CodeT5+                  | 16B    | 30.9      |      |                                                     | [paper](https://arxiv.org/abs/2305.07922) |
| 15   | LLaMA2-70B               | 70B    | 29.9      |      | [ckpt](https://hf.co/meta-llama/Llama-2-70b-hf)     | [paper](https://arxiv.org/abs/2307.09288) |
| 16   | CodeGen-16B-Mono         | 16B    | 29.3      | 35.3 |                                                     | [paper](https://arxiv.org/abs/2203.13474) |
| 17   | PaLM-540B                | 540B   | 26.2      | 36.8 |                                                     | [paper](https://arxiv.org/abs/2204.02311) |
| 18   | LLaMA-65B                | 65B    | 23.7      | 37.7 |                                                     | [paper](https://arxiv.org/abs/2302.13971) |
| 19   | CodeGeeX                 | 13B    | 22.9      | 24.4 |                                                     | [paper](https://arxiv.org/abs/2303.17568) |
| 20   | LLaMA-33B                | 33B    | 21.7      | 30.2 |                                                     | [paper](https://arxiv.org/abs/2302.13971) |
| 21   | CodeGen-16B-Multi        | 16B    | 18.3      | 20.9 |                                                     | [paper](https://arxiv.org/abs/2203.13474) |
| 22   | AlphaCode                | 1.1B   | 17.1      |      |                                                     | [paper](https://arxiv.org/abs/2203.07814) |


💡 Toolkit:
- [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness): A framework for the evaluation of autoregressive code generation language models.
- [multilingual-code-evals](https://huggingface.co/spaces/bigcode/multilingual-code-evals): Multilingual Code Models Evaluation.

## 📚 Paper

### ▶️ Pre-Training

1. **Evaluating Large Language Models Trained on Code** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2107.03374)] *Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto. et al.* 2021.07

2. **CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis** `ICLR23`
  
    [[Paper](https://arxiv.org/abs/2203.13474)] *Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, Caiming Xiong.* 2022.03

3. **SantaCoder: don't reach for the stars!** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2301.03988)] *Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff. et al.* 2023.01

4. **CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2303.17568)] *Qinkai Zheng, Xiao Xia, Xu Zou, Yuxiao Dong, Shan Wang, Yufei Xue, Zihan Wang, Lei Shen, Andi Wang, Yang Li, Teng Su, Zhilin Yang, Jie Tang.* 2023.03

5. **CodeGen2: Lessons for Training LLMs on Programming and Natural Languages** `ICLR23`
  
    [[Paper](https://arxiv.org/abs/2305.02309)] *Erik Nijkamp, Hiroaki Hayashi, Caiming Xiong, Silvio Savarese, Yingbo Zhou.* 2023.05

6. **StarCoder: may the source be with you!** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2305.06161)] *Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou. et al.* 2023.05

7. **CodeT5+: Open Code Large Language Models for Code Understanding and Generation** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2305.07922)] *Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi D.Q. Bui, Junnan Li, Steven C.H. Hoi.* 2023.05

8. **Textbooks Are All You Need** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2306.11644)] *Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi. et al.* 2023.06

9. **Code Llama: Open Foundation Models for Code** `Preprint`
  
    [[Paper](https://scontent-hkg4-1.xx.fbcdn.net/v/t39.2365-6/369856151_1754812304950972_1159666448927483931_n.pdf?_nc_cat=107&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=BnkB4kcpz5AAX9kJmnq&_nc_ht=scontent-hkg4-1.xx&oh=00_AfDk1DUUkNXLvMWb8tjPbZ_d1mvrtdl0AkTkhwf7svqHPw&oe=64ECB20F)] *Baptiste Rozière, Jonas Gehring, Fabian Gloeckle†, Sten Sootla†, Itai Gat. et al.* 2023.08



### ▶️ Instruction Tuning

1. **WizardCoder: Empowering Code Large Language Models with Evol-Instruct** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2306.08568)] *Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, Daxin Jiang.* 2023.07

2. **OctoPack: Instruction Tuning Code Large Language Models** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2308.07124)][[Repo](https://github.com/bigcode-project/octopack)] *Niklas Muennighoff, Qian Liu, Armel Zebaze, Qinkai Zheng, Binyuan Hui, Terry Yue Zhuo, Swayam Singh, Xiangru Tang, Leandro von Werra, Shayne Longpre.* 2023.08


### ▶️ Alignment with Feedback

1. **CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning** `NeurIPS22`
  
    [[Paper](https://arxiv.org/abs/2207.01780)] *Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, Steven C.H. Hoi.* 2022.07 

2. **Execution-based Code Generation using Deep Reinforcement Learning** `TMLR23`
  
    [[Paper](https://arxiv.org/abs/2301.13816)] *Parshin Shojaee, Aneesh Jain, Sindhu Tipirneni, Chandan K. Reddy.* 2023.01 

3. **RLTF: Reinforcement Learning from Unit Test Feedback** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2307.04349)] *Jiate Liu, Yiqin Zhu, Kaiwen Xiao, Qiang Fu, Xiao Han, Wei Yang, Deheng Ye.* 2023.07 

4. **PanGu-Coder2: Boosting Large Language Models for Code with Ranking Feedback** `Preprint`
  
    [[Paper](https://arxiv.org/abs/2307.14936)] *Bo Shen, Jiaxin Zhang, Taihong Chen, Daoguang Zan, Bing Geng, An Fu, Muhan Zeng, Ailun Yu, Jichuan Ji, Jingyang Zhao, Yuenan Guo, Qianxiang Wang.* 2023.07 


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



## 🙌 Contributors

<a href="https://github.com/huybery"><img src="https://avatars.githubusercontent.com/u/13436140?v=4"  width="50" /></a>
<a href="https://github.com/Yangjiaxi"><img src="https://avatars.githubusercontent.com/u/6203054?v=4"  width="50" /></a>
<a href="https://github.com/GanjinZero"><img src="https://avatars.githubusercontent.com/u/19466330?v=4"  width="50" /></a>

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
