# 👨‍💻 Awesome-Code-LLM [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![PR](https://img.shields.io/badge/PRs-Welcome-red)](https://img.shields.io/badge/PRs-Welcome-red) [![COMMIT](https://img.shields.io/github/last-commit/huybery/Awesome-Code-LLM?color=green)](https://img.shields.io/github/last-commit/huybery/Awesome-Code-LLM?color=green)

![](code-banner.png)

## 🧵 Table of Contents

- [👨‍💻 Awesome-Code-LLM   ](#-awesome-code-llm---)
  - [🧵 Table of Contents](#-table-of-contents)
  - [Evaluation](#evaluation)
    - [Toolkit](#toolkit)
  - [🙌 Contributors](#-contributors)
  - [Cite as](#cite-as)
  - [Acknowledgement](#acknowledgement)

## Evaluation

<p align="center"> Leaderboard (Order by HumanEval Pass@1) </p>

| Rank | Model                    | Open | HumanEval | MBPP | Source                                    |
|------|--------------------------|------|-----------|------|-------------------------------------------|
| 1    | GPT-4 + Relexion         |      | 91.0      | 77.1 | [paper](https://arxiv.org/abs/2303.11366) |
| 2    | GPT-4                    |      | 67.0      |      | [paper](https://arxiv.org/abs/2303.08774) |
| 3    | Pangu-Coder2-15B         |      | 61.6      |      | [paper](https://arxiv.org/abs/2307.14936) |
| 4    | WizardCoder-15B          | Y    | 57.3      | 51.8 | [paper](https://arxiv.org/abs/2306.08568) |
| 5    | GPT-3.5                  |      | 48.1      |      | [paper](https://arxiv.org/abs/2303.08774) |
| 6    | StarCoder-15B (Prompted) | Y    | 40.8      | 49.5 | [paper](https://arxiv.org/abs/2305.06161) |
| 7    | PaLM 2-S                 |      | 37.6      | 50.0 | [paper](https://arxiv.org/abs/2204.02311) |
| 8    | PaLM-Coder-540B          |      | 36.0      | 47.0 | [paper](https://arxiv.org/abs/2204.02311) |
| 9    | InstructCodeT5+          |      | 35.0      |      | [paper](https://arxiv.org/abs/2305.07922) |
| 10   | StarCoder-15B            | Y    | 33.6      | 52.7 | [paper](https://arxiv.org/abs/2305.06161) |
| 11   | Code-Cushman-001         |      | 33.5      | 45.9 | [paper](https://arxiv.org/abs/2107.03374) |
| 12   | CodeT5+                  | Y    | 30.9      |      | [paper](https://arxiv.org/abs/2305.07922) |
| 13   | LLaMA2-70B               | Y    | 29.9      |      | [paper](https://arxiv.org/abs/2307.09288) |
| 14   | CodeGen-16B-Mono         |      | 29.3      | 35.3 | [paper](https://arxiv.org/abs/2203.13474) |
| 15   | PaLM-540B                |      | 26.2      | 36.8 | [paper](https://arxiv.org/abs/2204.02311) |
| 16   | LLaMA-65B                | Y    | 23.7      | 37.7 | [paper](https://arxiv.org/abs/2302.13971) |
| 17   | CodeGeeX                 |      | 22.9      | 24.4 | [paper](https://arxiv.org/abs/2303.17568) |
| 18   | LLaMA-33B                | Y    | 21.7      | 30.2 | [paper](https://arxiv.org/abs/2302.13971) |
| 19   | CodeGen-16B-Multi        |      | 18.3      | 20.9 | [paper](https://arxiv.org/abs/2203.13474) |
| 20   | AlphaCode                |      | 17.1      |      | [paper](https://arxiv.org/abs/2203.07814) |


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
