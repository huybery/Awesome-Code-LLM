# 👨‍💻 Awesome-Code-LLM [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![PR](https://img.shields.io/badge/PRs-Welcome-red)](https://img.shields.io/badge/PRs-Welcome-red) [![COMMIT](https://img.shields.io/github/last-commit/huybery/Awesome-Code-LLM?color=green)](https://img.shields.io/github/last-commit/huybery/Awesome-Code-LLM?color=green)

![](code-banner.png)

## 🧵 Table of Contents

- [👨‍💻 Awesome-Code-LLM   ](#-awesome-code-llm---)
  - [🧵 Table of Contents](#-table-of-contents)
  - [Leaderboard](#leaderboard)
    - [Toolkit](#toolkit)
  - [🙌 Contributors](#-contributors)
  - [Cite as](#cite-as)
  - [Acknowledgement](#acknowledgement)

## Leaderboard

<p align="center"> <b>Leaderboard</b> (Sort by HumanEval Pass@1) </p>

| Rank | Model                    | Params | HumanEval | MBPP | HF                                                       | Paper                                     |
|------|--------------------------|--------|-----------|------|----------------------------------------------------------|-------------------------------------------|
| 1    | GPT-4 + Relexion         | ?      | 91.0      | 77.1 |                                                          | [paper](https://arxiv.org/abs/2303.11366) |
| 2    | GPT-4                    | ?      | 67.0      |      |                                                          | [paper](https://arxiv.org/abs/2303.08774) |
| 3    | Pangu-Coder2             | 15B    | 61.6      |      |                                                          | [paper](https://arxiv.org/abs/2307.14936) |
| 4    | WizardCoder-15B          | 15B    | 57.3      | 51.8 | [ckpt](https://hf.co/WizardLM/WizardCoder-15B-V1.0)      | [paper](https://arxiv.org/abs/2306.08568) |
| 5    | GPT-3.5                  | ?      | 48.1      |      |                                                          | [paper](https://arxiv.org/abs/2303.08774) |
| 6    | Code-Davinci-002         | ?      | 47.0      |      |                                                          | [paper](https://arxiv.org/abs/2107.03374) |
| 7    | StarCoder-15B (Prompted) | 15B    | 40.8      | 49.5 | [ckpt](https://huggingface.co/bigcode/starcoder)         | [paper](https://arxiv.org/abs/2305.06161) |
| 8    | PaLM 2-S                 | ?      | 37.6      | 50.0 |                                                          | [paper](https://arxiv.org/abs/2204.02311) |
| 9    | PaLM-Coder-540B          | 540B   | 36.0      | 47.0 |                                                          | [paper](https://arxiv.org/abs/2204.02311) |
| 10   | InstructCodeT5+          | 16B    | 35.0      |      |                                                          | [paper](https://arxiv.org/abs/2305.07922) |
| 11   | StarCoder-15B            | 15B    | 33.6      | 52.7 | [ckpt](https://huggingface.co/bigcode/starcoder)         | [paper](https://arxiv.org/abs/2305.06161) |
| 12   | Code-Cushman-001         | ?      | 33.5      | 45.9 |                                                          | [paper](https://arxiv.org/abs/2107.03374) |
| 13   | CodeT5+                  | 16B    | 30.9      |      |                                                          | [paper](https://arxiv.org/abs/2305.07922) |
| 14   | LLaMA2-70B               | 70B    | 29.9      |      | [ckpt](https://huggingface.co/meta-llama/Llama-2-70b-hf) | [paper](https://arxiv.org/abs/2307.09288) |
| 15   | CodeGen-16B-Mono         | 16B    | 29.3      | 35.3 |                                                          | [paper](https://arxiv.org/abs/2203.13474) |
| 16   | PaLM-540B                | 540B   | 26.2      | 36.8 |                                                          | [paper](https://arxiv.org/abs/2204.02311) |
| 17   | LLaMA-65B                | 65B    | 23.7      | 37.7 |                                                          | [paper](https://arxiv.org/abs/2302.13971) |
| 18   | CodeGeeX                 | 13B    | 22.9      | 24.4 |                                                          | [paper](https://arxiv.org/abs/2303.17568) |
| 19   | LLaMA-33B                | 33B    | 21.7      | 30.2 |                                                          | [paper](https://arxiv.org/abs/2302.13971) |
| 20   | CodeGen-16B-Multi        | 16B    | 18.3      | 20.9 |                                                          | [paper](https://arxiv.org/abs/2203.13474) |
| 21   | AlphaCode                | 1.1B   | 17.1      |      |                                                          | [paper](https://arxiv.org/abs/2203.07814) |


### Toolkit

- [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness): A framework for the evaluation of autoregressive code generation language models.
- [multilingual-code-evals](https://huggingface.co/spaces/bigcode/multilingual-code-evals): Multilingual Code Models Evaluation.

## 🙌 Contributors

<a href="https://github.com/huybery"><img src="https://avatars.githubusercontent.com/u/13436140?v=4"  width="50" /></a>
<a href="https://github.com/Yangjiaxi"><img src="https://avatars.githubusercontent.com/u/6203054?v=4"  width="50" /></a>

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

**[⬆ Back to ToC](#table-of-contents)**
