### **Code Completion**

**GitHub Java Corpus**｜Mining source code repositories at massive scale using language modeling ｜ 2013 ｜  EM/Acc./Edit sim

- present a curated corpus of 14,807 open-source Java projects from GitHub, comprising over 350 million lines of code (LOC)
    - filter by forked and duplicates
- N-Gram Language Model

**Py150 & JS150** | Probabilistic model for code with decision trees | 2016 | EM/Acc./Edit sim

- The key idea is to phrase the problem of learning a probabilistic model of code as learning a decision tree in a domain specific language over abstract syntax trees (AST)
    - [JavaScript datasets](http://www.srl.inf.ethz.ch/js150) 150k
    - [Python datasets](http://www.srl.inf.ethz.ch/py150) 150k

**DotPrompts** | Guiding Language Models of Code with Global Context using Monitors | 2023 | CR/NIM/ISM/PM

- propose monitor-guided decoding (MGD) where a monitor uses static analysis to guide the decoding.
- construct a **repository-level** dataset PRAGMATICCODE for method-completion in Java and evaluate MGD on it
    - From PRAGMATICCODE, we identify a set of method-level completion task instances, creating DOTPROMPTS as a method-level code completion benchmark.
    - Overall, PRAGMATICCODE consists of 100 repositories, and DOTPROMPTS consists of 1420 methods and 10538 dereference prompts.
    - From these files, we identify methods that aren’t class and object initializers and have ≥ 2 top-level statements spanning ≥ 7 lines of source code, to ensure sufficient complexity in target methods. The average number of lines of code in the ground truth completion in DOTPROMPTS is 12.7.

**LCC** ｜ LongCoder: A Long-Range Pre-trained Language Model for Code Completion ｜ 2023  ｜EM/Edit Sim

- Specifically, we construct our datasets from the [github-code](https://huggingface.co/datasets/%20codeparrot/github-code) dataset, which contains a vast number of code files sourced from GitHub with an open-source license that permits research use.
- Deduplicate & Pass Parsed AST & filted too short and too long
- Train 100k | Val 10k | Test 10k

**RepoBench** | RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems | 2023 | EM/Edit Sim

- **Retrieve** the most relevant code snippets from other files as cross-file context, **Predict** the next line of code with cross-file and in-file context, and **Handle** complex tasks that require a combination of both retrieval and next-line prediction.
- The first source of RepoBench is the github-code dataset, which consists of a vast collection of code files sourced from GitHub repositories under open-source licenses with a data cutoff date of March 16, 2022.
- comprises 10,345 Python and 14,956 Java historical repositories, serving as training data and are available for optional fine-tuning. Additionally, we have 1,075 Python and 594 Java new repositories from GitHub designated as test data for evaluation.
- Task Settings
    - `cross_file_first`: the next line of code utilizes content from a cross-file code snippet and it is its first usage within current file.
    - `cross_file_random`: the next line of code utilizes content from a cross-file code snippet and it is NOT its first usage within current file.
    - `in_file`: the next line of code does not utilize content from a cross-file code snippet.

### **Github**

**CommitAct** | Towards automatic generation of short summaries of commits | 2017 | Acc.

- The Data Set First, we obtained 967 commits from the work by "Dataset of developer-labeled commit messages". Second, we obtained all the commits from the top 1,000 popular **Java** projects in Github
- filtered the commit messages that are empty or have non-English letters. & Removing Special Commits (rollbacks and merge)
- 1.6M

**CommitGen** | A neural architecture for generating natural language descriptions from source code changes | 2017 |BLEU

- i) ease the comprehension of the dynamics of the system, which could be useful for debugging and repairing purposes
- ii) automate the documentation of source code changes.
- selecting 3 projects for each of the following languages: python, java, javascript and c++. For each project, we downloaded diff files and metadata of the full commit history.

**NNGen** ｜ Neural-machine-translation-based commit message generation: how far are we? ｜ 2018 ｜BLEU

**PtrGNCMsg** ｜Generating commit messages from diffs using pointer-generator network ｜ 2019 ｜BLEU/ROUGE

- The raw dataset is collected from the top 2,081 Java projects sorted in descending order of stars in GitHub using GitHub Developer REST API. **CommitAct** collects the top 1,000 projects, and we collect the top 1,0012,081 projects.

**CoDiSum**｜Commit message generation for source code changes｜2019 ｜BLEU/METEOR

- use the commonly-used dataset in this area, which was collected by **CommitAct** （509k）
- First, we remove the diff files that contain file changes other than .java files. Next, we apply standard NLP processing on the data. That is, we keep only the source code in the diff files, remove the punctuations and special symbols in the commit messages, tokenize diffs and commit messages, and remove the data that contains less than three words. Finally, we delete the duplicated diffs as there are some successive commits in a short period fixing the same bug with the same commit message.
- After pre-processing, we obtain 90, 661 pairs of diff, commit messages and randomly choose 75, 000 for training, 8, 000 for validation, and 7, 661 for testing.

**ATOM**｜ATOM: commit message generation based on abstract syntax tree and hybrid ranking ｜2019 ｜ BLEU

- including complete function-level code snippets of ∼160k commits from 56 java projects（ranked by “star numbers“）. We clean the benchmark by filtering out meaningless (e.g., empty, non-ASCII, merge) commits

**CommitBERT**｜CommitBERT: Commit message generation using pre-trained programming language model ｜2021 ｜BLEU

- collect code modification and corresponding commit messages in Github for six languages (Python, PHP, Go, Java, JavaScript, and Ruby) and release a wellorganized 345K pair dataset
- focuses only on the added and deleted lines in git diff

**MCMD** ｜ On the evaluation of commit message generation models: An experimental study ｜2021 ｜B-Moses/B-Norm/B-CC

- For each language, we collected commits before 2021 from the top 100 starred projects on GitHub.
- To balance the size of data in each programming language so that we can fairly compare the performance of models in different programming language in subsequent experiments, we randomly sampled and retained 450,000 commits for each language.
- MCMD contains the complete information of commits, including not only code diffs and commit messages, but also RepoFullname, SHA, and timestamp.
- Related work good summary: COMMIT MESSAGE GENERATION

**CoRec** |Context-aware retrieval-based deep commit message generation |2021 |BLEU

- **CommitAct** only collected top 1,000 Java projects from Github and their dataset represents only 1.75% out of 2 million commits in GitHub. we collected the top 10,000 repositories (ordered by the number of stars) which were created between January 2012 and December 2018 in Github.

**ExGroFi** | Delving into commit-issue correlation to enhance commit message generation models | 2023 | BLEU/ROUGE/CIDEr

- The dataset consists of an unlabeled commit-issue parallel part and a labeled part in which each example is provided with human-annotated rational information in the issue.

**CommitChronicle** | From commit message generation to history-aware commit message completion|2023| Resolve Rate/Recall

- Since the existing datasets lack historical data, we collect and share a novel dataset called CommitChronicle, containing 10.7M commits across 20 programming languages.
- used the GitHub Search tool on January 25th, 2023 to select specific repositories for subsequent data mining.

**SWE-bench** ｜ Swe-bench: Can language models resolve real-world GitHub issues? ｜2023 ｜Resolve Rate/Recall

- an evaluation framework including 2,294 software engineering problems drawn from real GitHub issues and corresponding pull requests across 12 popular Python repositories.

**Commitbench** | Commitbench: A benchmark for commit message generation | 2023 | BLEU/METEOR/ROUGE

- existing datasets exhibit various problems, such as the quality of the commit selection, small sample sizes, duplicates, privacy issues, and missing licenses for redistribution.
- We sample commits from diverse projects with licenses that permit redistribution and apply our filtering and dataset enhancements to improve the quality of generated commit messages.

**DevBench** | DevBench: A Comprehensive Benchmark for Software Development | 2014 | pass@k

- a comprehensive benchmark that evaluates LLMs across various stages of the software development lifecycle, including software design, environment setup, implementation, acceptance testing, and unit testing.
- contains a collection of 22 curated repositories, spanning across four widely-used programming languages (Python, C/C++, Java, JavaScript) and a diverse range of domains

### Code Summarization

**CODE-NN** | Summarizing source code using a neural attention model | 2016 | BLEU

- collected data from StackOverflow
- For the final dataset, we retain 66,015 C# (title, query) pairs and 32,337 SQL pairs that are classified as clean

**DeepCom ｜**Deep code comment generation ｜ 2018 ｜BLEU

- This paper proposes a new approach named DeepCom to automatically generate code comments for Java methods.
- get 69,708 ⟨ Java method, comment⟩ pairs

**TL-CodeSum** | Summarizing source code with transferred api knowledge | 2018 | BLEU

- The two datasets are both collected from GitHub. The API sequences summarization dataset contains Java projects from 2009 to 2014 and is used to learn API knowledge.
- At last, we get 340,922 pairs of 〈API sequence, summary〉 for API knowledge learning in API sequences summarization task and 69,708 pairs of 〈 API sequence, code, summary〉 for code summarization task.

**CodeSearchNet** | Codesearchnet challenge: Evaluating the state of semantic code search | 2019 | MRR

- CodeSearchNet Corpus: The corpus contains about 6 million functions from open-source code spanning six programming languages (Go, Java, JavaScript, PHP, Python, and Ruby). The CodeSearchNet Corpus also contains automatically generated query-like natural language for 2 million functions, obtained from mechanically scraping and preprocessing associated function documentation.
- CodeSearchNet Challenge: which consists of 99 natural language queries with about 4k expert relevance annotations of likely results from CodeSearchNet Corpus.

**HumanEvalPack** | Octopack: Instruction tuning code large language models | 2023 | BLEU

- We further introduce HUMANEVALPACK, expanding the HumanEval benchmark to a total of 3 coding tasks (Code Repair, Code Explanation, Code Synthesis) across 6 languages (Python, JavaScript, Java, Go, C++, Rust).

### Code Debug

**QuixBugs** ｜ A comprehensive study of automatic program repair on the quixbugs benchmark ｜ 2017 ｜Pass Rate

- a benchmark of 40 bugs from 40 classic algorithms such as sorting algorithms of bucket sort, merge sort and quick sort. All bugs of QuixBugs were collected from the Quixey Challenges, which consisted of giving human developers one minute to fix one program with a bug on a single line. (Java and Python)

**EvalGPTFix** | A Critical Review of Large Language Model on Software Engineering: An Example from ChatGPT and Automated Program Repair | 2023 | Pass Rate

Raw Data Collection. We first crawl all the Java submissions of AtCoder programming contests starting from 2023. We focus on Java languages as it is the most targeted language in the APR community.

**DebugBench** | DebugBench: Evaluating Debugging Capability of Large Language Models | 2024 | Pass Rate

- an LLM debugging benchmark consisting of 4,253 instances. It covers four major bug categories and 18 minor types in C++, Java, and Python.
- collect code snippets from the LeetCode community, implant bugs into source data with GPT-4, and assure rigorous quality checks.

### Question Answering

**CodeQA** ｜ CodeQA: A question answering dataset for source code comprehension ｜2021 ｜BLEU/EM/F1

- CodeQA contains a Java dataset with 119,778 question-answer pairs and a Python dataset with 70,085 question-answer pairs
- To obtain natural and faithful questions and answers, we implement syntactic rules and semantic analysis to transform code comments into question-answer pairs.

**CodeQueries**｜CodeQueries: A Dataset of Semantic Queries over Code ｜ 2022 ｜EM/Acc./Recall

- CodeQueries, of semantic queries over Python code. Compared to the existing datasets, in CodeQueries, the queries are about code semantics, the context is file level and the answers are code spans.

### Code Editing

**EditEval** ｜ Instructcoder: Empowering language models for code editing ｜2023 ｜Acc.

- InstructCoder, the first instructiontuning dataset designed to adapt LLMs for general-purpose code editing, containing highdiversity code-editing tasks such as comment insertion, code optimization, and code refactoring. It consists of over 114,000 instructioninput-output triplets and covers multiple distinct code editing scenarios.
- human-written execution-based benchmark dubbed EditEval
- It contains various types of code edits adapted from Github commits and existing datasets.