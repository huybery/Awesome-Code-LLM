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
- [🚀 Top Code LLMs](#-top-code-llms)
- [💡 Evaluation Toolkit:](#-evaluation-toolkit)
- [🚀 Awesome Code Leaderboard](#-awesome-code-leaderboard)
- [📚 Awesome Papers](#-awesome-papers)
  - [▶️ Awesome Code Pre-Training Papers](#️ -awesome-code-pre-training-papers)
  - [▶️ Awesome Code Instruction-Tuning Papers](#️ -awesome-code-instruction-tuning-papers)
  - [▶️ Awesome Code Alignment Papers](#️ -awesome-code-alignment-papers)
  - [▶️ Awesome Code Prompting Papers](#️ -awesome-code-prompting-papers)
  - [▶️ Awesome Code Benchmark \& Evaluation Papers](#️ -awesome-code-benchmark--evaluation-papers)
- [🙌 Contributors](#-contributors)
- [Cite as](#cite-as)
- [Acknowledgement](#acknowledgement)
- [Star History](#star-history)


## 🚀 Top Code LLMs
###### Sort by HumanEval Pass@1

| Rank | Model                                                                                           | Params  | HumanEval | MBPP | Source                                                     |
|------|-------------------------------------------------------------------------------------------------|---------|-----------|------|------------------------------------------------------------|
| 1    | o1-mini-2024-09-12                                                                              | -       | 97.6      | 93.9 | [paper](https://arxiv.org/abs/2409.12186)                  |
| 2    | o1-preview-2024-09-12                                                                           | -       | 95.1      | 93.4 | [paper](https://arxiv.org/abs/2409.12186)                  |
| 3    | [Qwen2.5-Coder-32B-Instruct]((https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct))          | 32B     | 92.7      | 90.2 | [github](https://github.com/QwenLM/Qwen2.5-Coder)          |
| 4    | Claude-3.5-Sonnet-20241022                                                                      | -       | 92.1      | 91.0 | [paper](https://arxiv.org/abs/2409.12186)                  |
| 5    | GPT-4o-2024-08-06                                                                               | -       | 92.1      | 86.8 | [paper](https://arxiv.org/abs/2409.12186)                  |
| 6    | [Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct)            | 14B     | 89.6      | 86.2 | [github](https://github.com/QwenLM/Qwen2.5-Coder)          |
| 7    | Claude-3.5-Sonnet-20240620                                                                      | -       | 89.0      | 87.6 | [paper](https://arxiv.org/abs/2409.12186)                  |
| 8    | GPT-4o-mini-2024-07-18                                                                          | -       | 87.8      | 86.0 | [paper](https://arxiv.org/abs/2409.12186)                  |
| 9    | [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)              | 7B      | 88.4      | 83.5 | [github](https://github.com/QwenLM/Qwen2.5-Coder)          |
| 10   | [DS-Coder-V2-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct)           | 21/236B | 85.4      | 89.4 | [github](https://github.com/deepseek-ai/DeepSeek-Coder-V2) |
| 11   | [Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct)              | 3B      | 84.1      | 73.6 | [github](https://github.com/QwenLM/Qwen2.5-Coder)          |
| 12   | [DS-Coder-V2-Lite-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) | 2.4/16B | 81.1      | 82.8 | [github](https://github.com/deepseek-ai/DeepSeek-Coder-V2) |
| 13   | [CodeQwen1.5-7B-Chat](https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat)                          | 7B      | 83.5      | 70.6 | [github](https://github.com/QwenLM/CodeQwen1.5)            |
| 14   | [DeepSeek-Coder-33B-Instruct](https://hf.co/deepseek-ai/deepseek-coder-33b-instruct)            | 33B     | 79.3      | 70.0 | [github](https://github.com/deepseek-ai/DeepSeek-Coder)    |
| 15   | [DeepSeek-Coder-6.7B-Instruct](https://hf.co/deepseek-ai/deepseek-coder-6.7b-instruct)          | 6.7B    | 78.6      | 65.4 | [github](https://github.com/deepseek-ai/DeepSeek-Coder)    |
| 16   | GPT-3.5-Turbo                                                                                   | -       | 76.2      | 70.8 | [github](https://github.com/deepseek-ai/DeepSeek-Coder)    |
| 17   | [CodeLlama-70B-Instruct](https://huggingface.co/meta-llama/CodeLlama-70b-Instruct-hf)           | 70B     | 72.0      | 77.8 | [paper](https://arxiv.org/abs/2308.12950)                  |
| 18   | [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)          | 1.5B    | 70.7      | 69.2 | [github](https://github.com/QwenLM/Qwen2.5-Coder)          |
| 19   | [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)          | 0.5B    | 61.6      | 52.4 | [github](https://github.com/QwenLM/Qwen2.5-Coder)          |
| 20   | Pangu-Coder2                                                                                    | 15B     | 61.6      |      | [paper](https://arxiv.org/abs/2307.14936)                  |
| 21   | [WizardCoder-15B](https://hf.co/WizardLM/WizardCoder-15B-V1.0)                                  | 15B     | 57.3      | 51.8 | [paper](https://arxiv.org/abs/2306.08568)                  |
| 22   | CodeQwen1.5-7B                                                                                  | 7B      | 51.8      | 61.8 | [github](https://github.com/QwenLM/CodeQwen1.5)            |
| 23   | [CodeLlama-34B-Instruct](https://huggingface.co/meta-llama/CodeLlama-34b-Instruct-hf)           | 34B     | 48.2      | 61.1 | [paper](https://arxiv.org/abs/2308.12950)                  |
| 24   | Code-Davinci-002                                                                                | -       | 47.0      |      | [paper](https://arxiv.org/abs/2107.03374)                  |
| 25   | [StarCoder2-15B-Instruct-v0.1](https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1)     | 15B     | 67.7      | 78.0 | [paper](https://arxiv.org/abs/2305.06161)                  |
&nbsp;

## 💡 Evaluation Toolkit:

- [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness): A framework for the evaluation of autoregressive code generation language models.
- [code-eval](https://github.com/abacaj/code-eval): A framework for the evaluation of autoregressive code generation language models on HumanEval.
&nbsp;

## 🚀 Awesome Code Leaderboard
| Leaderboard                          | Access                                                                            |
|--------------------------------------|-----------------------------------------------------------------------------------|
| Evalperf Leaderboard                 | [[Source](https://evalplus.github.io/evalperf.html)]                              |
| Aider Code Editing Leaderboard       | [[Source](https://aider.chat/docs/leaderboards/)]                                 |
| BigCodeBench Leaderboard             | [[Source](https://bigcode-bench.github.io)]                                       |
| LiveCodeBench Leaderboard            | [[Source](https://livecodebench.github.io/leaderboard.html)]                      |
| Big Code Models Leaderboard          | [[Source](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)]      |
| BIRD                                 | [[Source](https://bird-bench.github.io)]                                          |
| CanAiCode Leaderboard                | [[Source](https://huggingface.co/spaces/mike-ravkine/can-ai-code-results)]        |
| Coding LLMs Leaderboard              | [[Source](https://leaderboard.tabbyml.com)]                                       |
| CRUXEval Leaderboard                 | [[Source](https://crux-eval.github.io/leaderboard.html)]                          |
| EvalPlus Leaderboard                 | [[Source](https://evalplus.github.io/leaderboard.html)]                           |
| HumanEval.jl                         | [[Source](https://github.com/01-ai/HumanEval.jl)]                                 |
| InfiCoder-Eval                       | [[Source](https://infi-coder.github.io/inficoder-eval)]                           |
| InterCode                            | [[Source](https://intercode-benchmark.github.io)]                                 |
| Program Synthesis Models Leaderboard | [[Source](https://accubits.com/open-source-program-synthesis-models-leaderboard)] |
| Spider                               | [[Source](https://yale-lily.github.io/spider)]                                    |
&nbsp;


## 📚 Awesome Papers

### Awesome Code Pre-Training Papers
| Title                                                                                                                                                                                                                                                  | Venue      | Date      | Code                                                       | Resources                                                                         |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-----------|------------------------------------------------------------|-----------------------------------------------------------------------------------|
| ![Star](https://img.shields.io/github/stars/OpenCoder-llm/OpenCoder-llm.svg?style=social&label=Star) <br> [**OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models**](https://arxiv.org/abs/2411.04905) <br>                            | `Preprint` | `2024.11` | [Github](https://github.com/OpenCoder-llm/OpenCoder-llm)   | [HF](https://huggingface.co/infly/OpenCoder-8B-Instruct)                          |
| ![Star](https://img.shields.io/github/stars/QwenLM/Qwen2.5-Coder.svg?style=social&label=Star) <br> [**Qwen2.5-Coder Technical Report**](https://arxiv.org/abs/2409.12186) <br>                                                                         | `Preprint` | `2024.09` | [Github](https://github.com/QwenLM/Qwen2.5-Coder)          | [HF](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)                      |
| ![Star](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-Coder-V2.svg?style=social&label=Star) <br> [**DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence**](https://arxiv.org/abs/2406.11931) <br>          | `Preprint` | `2024.06` | [Github](https://github.com/deepseek-ai/DeepSeek-Coder-V2) | [HF](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct)               |
| ![Star](https://img.shields.io/github/stars/bigcode-project/starcoder2.svg?style=social&label=Star) <br> [**StarCoder 2 and The Stack v2: The Next Generation**](https://arxiv.org/abs/2402.19173) <br>                                                | `Preprint` | `2024.02` | [Github](https://github.com/bigcode-project/starcoder2)    | [HF](https://huggingface.co/bigcode)                                              |
| ![Star](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-Coder.svg?style=social&label=Star) <br> [**DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence**](https://arxiv.org/abs/2401.14196) <br> | `Preprint` | `2024.01` | [Github](https://github.com/deepseek-ai/DeepSeek-Coder)    | [HF](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct)              |
| ![Star](https://img.shields.io/github/stars/meta-llama/codellama.svg?style=social&label=Star) <br> [**Code Llama: Open Foundation Models for Code**](https://arxiv.org/abs/2308.12950) <br>                                                            | `Preprint` | `2023.08` | [Github](https://github.com/meta-llama/codellama)          | [HF](https://huggingface.co/meta-llama/CodeLlama-7b-hf)                           |
| [**Textbooks Are All You Need**](https://arxiv.org/abs/2306.11644) <br>                                                                                                                                                                                | `Preprint` | `2023.06` | -                                                          | [HF](https://huggingface.co/microsoft/phi-1)                                      |
| ![Star](https://img.shields.io/github/stars/salesforce/CodeT5.svg?style=social&label=Star) <br> [**CodeT5+: Open Code Large Language Models for Code Understanding and Generation**](https://arxiv.org/abs/2305.07922) <br>                            | `Preprint` | `2023.05` | [Github](https://github.com/salesforce/CodeT5)             | [HF](https://huggingface.co/Salesforce/codet5p-16b)                               |
| ![Star](https://img.shields.io/github/stars/bigcode-project/starcoder.svg?style=social&label=Star) <br> [**StarCoder: may the source be with you!**](https://arxiv.org/abs/2305.06161) <br>                                                            | `Preprint` | `2023.05` | [Github](https://github.com/bigcode-project/starcoder)     | [HF](https://huggingface.co/bigcode/starcoder)                                    |
| ![Star](https://img.shields.io/github/stars/salesforce/CodeGen.svg?style=social&label=Star) <br> [**CodeGen2: Lessons for Training LLMs on Programming and Natural Languages**](https://arxiv.org/abs/2305.02309) <br>                                 | `ICLR23`   | `2023.05` | [Github](https://github.com/salesforce/CodeGen)            | [HF](https://huggingface.co/Salesforce/codegen25-7b-multi_P)                      |
| ![Star](https://img.shields.io/github/stars/THUDM/CodeGeeX.svg?style=social&label=Star) <br> [**CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X**](https://arxiv.org/abs/2303.17568) <br>               | `Preprint` | `2023.03` | [Github](https://github.com/THUDM/CodeGeeX)                | [HF](https://huggingface.co/collections/THUDM/codegeex4-6694e777e98246f00632fcf1) |
| [**SantaCoder: don't reach for the stars!**](https://arxiv.org/abs/2301.03988) <br>                                                                                                                                                                    | `Preprint` | `2023.01` | -                                                          | [HF](https://huggingface.co/bigcode/santacoder)                                   |
| ![Star](https://img.shields.io/github/stars/salesforce/CodeGen.svg?style=social&label=Star) <br> [**CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis**](https://arxiv.org/abs/2203.13474) <br>                         | `ICLR'23`  | `2022.03` | [Github](https://github.com/salesforce/CodeGen)            | [HF](https://huggingface.co/Salesforce/codegen25-7b-multi_P)                      |
| ![Star](https://img.shields.io/github/stars/openai/human-eval.svg?style=social&label=Star) <br> [**Evaluating Large Language Models Trained on Code**](https://arxiv.org/abs/2107.03374) <br>                                                          | `Preprint` | `2021.07` | [Github](https://github.com/openai/human-eval)             | -                                                                                 |
&nbsp;

### Awesome Code Instruction-Tuning Papers
| Title                                                                                                                                                                                                                                                | Venue      | Date      | Code                                                  | Resources                                                      |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-----------|-------------------------------------------------------|----------------------------------------------------------------|
| ![Star](https://img.shields.io/github/stars/ise-uiuc/magicoder.svg?style=social&label=Star) <br> [**Magicoder: Source Code Is All You Need**](https://arxiv.org/abs/2312.02120) <br>                                                                 | `ICML'24`  | `2023.12` | [Github](https://github.com/ise-uiuc/magicoder)       | [HF](https://huggingface.co/ise-uiuc/Magicoder-DS-6.7B)        |
| ![Star](https://img.shields.io/github/stars/bigcode-project/octopack.svg?style=social&label=Star) <br> [**OctoPack: Instruction Tuning Code Large Language Models**](https://arxiv.org/abs/2308.07124) <br>                                          | `ICLR'24`  | `2023.08` | [Github](https://github.com/bigcode-project/octopack) | [HF](https://huggingface.co/bigcode/octocoder)                 |
| ![Star](https://img.shields.io/github/stars/nlpxucan/WizardLM.svg?style=social&label=Star) <br> [**WizardCoder: Empowering Code Large Language Models with Evol-Instruct**](https://arxiv.org/abs/2306.08568) <br>                                   | `Preprint` | `2023.07` | [Github](https://github.com/nlpxucan/WizardLM)        | [HF](https://huggingface.co/WizardLMTeam/WizardCoder-15B-V1.0) |
| ![Star](https://img.shields.io/github/stars/sahil280114/codealpaca.svg?style=social&label=Star) <br> [**Code Alpaca: An Instruction-following LLaMA Model trained on code generation instructions**](https://github.com/sahil280114/codealpaca) <br> | `Preprint` | `2023.xx` | [Github](https://github.com/sahil280114/codealpaca)   | [HF](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) |
&nbsp;


### Awesome Code Alignment Papers
| Title                                                                                                                                                                                                                                    | Venue        | Date      | Code                                                          | Resources |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|-----------|---------------------------------------------------------------|-----------|
| [**PLUM: Preference Learning Plus Test Cases Yields Better Code Language Models**](https://arxiv.org/abs/2406.06887) <br>                                                                                                                | `Preprint`   | `2024.06` | -                                                             | -         |
| [**PanGu-Coder2: Boosting Large Language Models for Code with Ranking Feedback**](https://arxiv.org/abs/2307.14936) <br>                                                                                                                 | `Preprint`   | `2023.07` | -                                                             | -         |
| ![Star](https://img.shields.io/github/stars/Zyq-scut/RLTF.svg?style=social&label=Star) <br> [**RLTF: Reinforcement Learning from Unit Test Feedback**](https://arxiv.org/abs/2307.04349) <br>                                            | `Preprint`   | `2023.07` | [Github](https://github.com/Zyq-scut/RLTF)                    | -         |
| ![Star](https://img.shields.io/github/stars/reddy-lab-code-research/PPOCoder.svg?style=social&label=Star) <br> [**Execution-based Code Generation using Deep Reinforcement Learning**](https://arxiv.org/abs/2301.13816) <br>            | `TMLR'23`    | `2023.01` | [Github](https://github.com/reddy-lab-code-research/PPOCoder) | -         |
| ![Star](https://img.shields.io/github/stars/salesforce/CodeRL.svg?style=social&label=Star) <br> [**CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning**](https://arxiv.org/abs/2207.01780) <br> | `NeurIPS'22` | `2022.07` | [Github](https://github.com/salesforce/CodeRL)                | -         |
&nbsp;

### Awesome Code Prompting Papers
| Title                                                                                                                                                                                                                                                 | Venue      | Date      | Code                                                                   | Resources |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-----------|------------------------------------------------------------------------|-----------|
| ![Star](https://img.shields.io/github/stars/YerbaPage/MGDebugger.svg?style=social&label=Star) <br> [**From Code to Correctness: Closing the Last Mile of Code Generation with Hierarchical Debugging**](https://arxiv.org/abs/2410.01215) <br>        | `Preprint` | `2024.10` | [Github](https://github.com/YerbaPage/MGDebugger)                      | -         |
| ![Star](https://img.shields.io/github/stars/FloridSleeves/LLMDebugger.svg?style=social&label=Star) <br> [**Debug like a Human: A Large Language Model Debugger via Verifying Runtime Execution Step-by-step**](https://arxiv.org/abs/2402.16906) <br> | `ACL'24`   | `2024.02` | [Github](https://github.com/FloridSleeves/LLMDebugger)                 | -         |
| [**SelfEvolve: A Code Evolution Framework via Large Language Models**](https://arxiv.org/abs/2306.02907) <br>                                                                                                                                         | `Preprint` | `2023.06` | -                                                                      | -         |
| ![Star](https://img.shields.io/github/stars/theoxo/self-repair.svg?style=social&label=Star) <br> [**Demystifying GPT Self-Repair for Code Generation**](https://arxiv.org/abs/2306.09896) <br>                                                        | `ICLR'24`  | `2023.06` | [Github](https://github.com/theoxo/self-repair)                        | -         |
| [**Teaching Large Language Models to Self-Debug**](https://arxiv.org/abs/2304.05128) <br>                                                                                                                                                             | `ICLR'24`  | `2023.06` | -                                                                      | -         |
| ![Star](https://img.shields.io/github/stars/niansong1996/lever.svg?style=social&label=Star) <br> [**LEVER: Learning to Verify Language-to-Code Generation with Execution**](https://arxiv.org/abs/2302.08468) <br>                                    | `ICML'23`  | `2023.02` | [Github](https://github.com/niansong1996/lever)                        | -         |
| ![Star](https://img.shields.io/github/stars/facebookresearch/coder_reviewer_reranking.svg?style=social&label=Star) <br> [**Coder Reviewer Reranking for Code Generation**](https://arxiv.org/abs/2211.16490) <br>                                     | `ICML'23`  | `2022.11` | [Github](https://github.com/facebookresearch/coder_reviewer_reranking) | -         |
| ![Star](https://img.shields.io/github/stars/microsoft/CodeT.svg?style=social&label=Star) <br> [**CodeT: Code Generation with Generated Tests**](https://arxiv.org/abs/2207.10397) <br>                                                                | `ICLR'23`  | `2022.07` | [Github](https://github.com/microsoft/CodeT)                           | -         |
&nbsp;

### Awesome Code Benchmark & Evaluation Papers
| Dataset         | Title                                                                                                                                                                                                                                                           | Venue        | Date      | Code                                                                                    | Resources                                                                |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|-----------|-----------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `Evalperf`      | ![Star](https://img.shields.io/github/stars/evalplus/evalplus.svg?style=social&label=Star) <br> [**Evaluating Language Models for Efficient Code Generation**](https://arxiv.org/abs/2408.06450) <br>                                                           | `COLM'24`    | `2024.08` | [Github](https://github.com/evalplus/evalplus)                                          | [HF](https://huggingface.co/evalplus)                                    |
| `LiveCodeBench` | ![Star](https://img.shields.io/github/stars/LiveCodeBench/LiveCodeBench.svg?style=social&label=Star) <br> [**LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code**](https://arxiv.org/abs/2403.07974) <br>              | `Preprint`   | `2024.03` | [Github](https://github.com/LiveCodeBench/LiveCodeBench)                                | [HF](https://huggingface.co/datasets/livecodebench/code_generation_lite) |
| `DevBench`      | ![Star](https://img.shields.io/github/stars/open-compass/DevBench.svg?style=social&label=Star) <br> [**DevBench: A Comprehensive Benchmark for Software Development**](https://arxiv.org/abs/2403.08604) <br>                                                   | `Preprint`   | `2024.03` | [Github](https://github.com/open-compass/DevBench)                                      | -                                                                        |
| `SWE-bench`     | ![Star](https://img.shields.io/github/stars/princeton-nlp/SWE-bench.svg?style=social&label=Star) <br> [**SWE-bench: Can Language Models Resolve Real-World GitHub Issues?**](https://arxiv.org/abs/2310.06770) <br>                                             | `ICLR'24`    | `2024.03` | [Github](https://github.com/princeton-nlp/SWE-bench)                                    | [HF](https://huggingface.co/datasets/princeton-nlp/SWE-bench)            |
| `CrossCodeEval` | ![Star](https://img.shields.io/github/stars/amazon-science/cceval.svg?style=social&label=Star) <br> [**CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion**](https://arxiv.org/abs/2306.03091) <br>                             | `NeurIPS'23` | `2023.11` | [Github](https://github.com/amazon-science/cceval)                                      | -                                                                        |
| `RepoCoder`     | ![Star](https://img.shields.io/github/stars/microsoft/CodeT.svg?style=social&label=Star) <br> [**Repository-Level Code Completion Through Iterative Retrieval and Generation**](https://arxiv.org/abs/2306.03091) <br>                                          | `EMNLP'23`   | `2023.10` | [Github](https://github.com/microsoft/CodeT/tree/main/RepoCoder)                        | -                                                                        |
| `LongCoder`     | ![Star](https://img.shields.io/github/stars/microsoft/CodeBERT.svg?style=social&label=Star) <br> [**LongCoder: A Long-Range Pre-trained Language Model for Code Completion**](https://arxiv.org/abs/2306.14893) <br>                                            | `ICML'23`    | `2023.10` | [Github](https://github.com/microsoft/CodeBERT)                                         | -                                                                        |
| -               | [**Can ChatGPT replace StackOverflow? A Study on Robustness and Reliability of Large Language Model Code Generation**](https://arxiv.org/abs/2308.10335) <br>                                                                                                   | `Preprint`   | `2023.08` | -                                                                                       | -                                                                        |
| `BioCoder`      | ![Star](https://img.shields.io/github/stars/gersteinlab/BioCoder.svg?style=social&label=Star) <br> [**BioCoder: A Benchmark for Bioinformatics Code Generation with Large Language Models**](https://arxiv.org/abs/2308.16458) <br>                             | `ISMB'24`    | `2023.08` | [Github](https://github.com/gersteinlab/BioCoder)                                       | -                                                                        |
| `RepoBench`     | ![Star](https://img.shields.io/github/stars/Leolty/repobench.svg?style=social&label=Star) <br> [**RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems**](https://arxiv.org/abs/2306.03091) <br>                                               | `ICLR'24`    | `2023.06` | [Github](https://github.com/Leolty/repobench)                                           | [HF](https://huggingface.co/datasets/tianyang/repobench_python_v1.1)     |
| `Evalplus`      | ![Star](https://img.shields.io/github/stars/evalplus/evalplus.svg?style=social&label=Star) <br> [**Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation**](https://arxiv.org/abs/2305.01210) <br> | `NeurIPS'23` | `2023.05` | [Github](https://github.com/evalplus/evalplus)                                          | [HF](https://huggingface.co/evalplus)                                    |
| `Coeditor`      | ![Star](https://img.shields.io/github/stars/MrVPlusOne/Coeditor.svg?style=social&label=Star) <br> [**Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing**](https://arxiv.org/abs/2305.18584) <br>                                        | `ICLR'24`    | `2023.05` | [Github](https://github.com/MrVPlusOne/Coeditor)                                        | -                                                                        |
| `DS-1000`       | ![Star](https://img.shields.io/github/stars/xlang-ai/DS-1000.svg?style=social&label=Star) <br> [**DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation**](https://arxiv.org/abs/2211.11501) <br>                                          | `ICML'23`    | `2022.11` | [Github](https://github.com/xlang-ai/DS-1000)                                           | [HF](https://huggingface.co/datasets/xlangai/DS-1000)                    |
| `MultiPL-E`     | ![Star](https://img.shields.io/github/stars/nuprl/MultiPL-E.svg?style=social&label=Star) <br> [**MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation**](https://arxiv.org/abs/2208.08227) <br>                                 | `Preprint`   | `2022.08` | [Github](https://github.com/nuprl/MultiPL-E)                                            | [HF](https://huggingface.co/datasets/xlangai/DS-1000)                    |
| `MBPP`          | ![Star](https://img.shields.io/github/stars/google-research/google-research.svg?style=social&label=Star) <br> [**Program Synthesis with Large Language Models**](https://arxiv.org/abs/2108.07732) <br>                                                         | `Preprint`   | `2021.08` | [Github](https://github.com/google-research/google-research/blob/master/mbpp/README.md) | [HF](https://huggingface.co/datasets/nuprl/MultiPL-E)                    |
| `APPS`          | ![Star](https://img.shields.io/github/stars/hendrycks/apps.svg?style=social&label=Star) <br> [**Measuring Coding Challenge Competence With APPS**](https://arxiv.org/abs/2105.09938) <br>                                                                       | `NeurIPS'21` | `2021.05` | [Github](https://github.com/hendrycks/apps)                                             | [HF](https://huggingface.co/datasets/codeparrot/apps)                    |
&nbsp;

## 🙌 Contributors

<a href="https://github.com/huybery"><img src="https://avatars.githubusercontent.com/u/13436140?v=4"  width="50" /></a>
<a href="https://github.com/Yangjiaxi"><img src="https://avatars.githubusercontent.com/u/6203054?v=4"  width="50" /></a>
<a href="https://github.com/GanjinZero"><img src="https://avatars.githubusercontent.com/u/19466330?v=4"  width="50" /></a>
<a href="https://github.com/TyDunn"><img src="https://avatars.githubusercontent.com/u/13314504?v=4"  width="50" /></a>
<a href="https://github.com/Hambaobao"><img src="https://avatars.githubusercontent.com/u/48345096?v=4"  width="50" /></a>

This is an active repository and your contributions are always welcome! If you have any question about this opinionated list, do not hesitate to contact me `huybery@gmail.com`.
&nbsp;

## Cite as

```
@software{awesome-code-llm,
  author = {Binyuan Hui, Lei Zhang},
  title = {An awesome and curated list of best code-LLM for research},
  howpublished = {\url{https://github.com/huybery/Awesome-Code-LLM}},
  year = 2023,
}
```
&nbsp;

## Acknowledgement

This project is inspired by [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM).
&nbsp;

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=huybery/Awesome-Code-LLM&type=Date)](https://star-history.com/#huybery/Awesome-Code-LLM&Date)


**[⬆ Back to ToC](#table-of-contents)**