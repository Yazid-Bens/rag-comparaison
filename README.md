# RAG Approaches Comparison

Research code and experimental results for comparing nine
Retrieval-Augmented Generation (RAG) implementations using a local
Qwen3-8B model on HotpotQA, MuSiQue, and ARCD.

The benchmark examines answer quality, retrieval effectiveness,
model-call counts, and retrieval/generation latency.

## Compared methods

| Method | Identifier in code and CSV files |
|---|---|
| Naive RAG | `Naive` |
| Hybrid RAG | `Hybrid` |
| HyDE | `HyDE` |
| Query Expansion | `QueryExpansion` |
| Corrective RAG | `Corrective` |
| GraphRAG | `GraphRAG` |
| Agentic RAG v1 | `Agentic_v1` |
| Agentic RAG v2 | `Agentic` |
| Adaptive RAG | `Adaptive` |

These are the project's implementations of the approaches.
The method names do not imply exact reproductions of every component
of the original published systems.

## Datasets

| Dataset | Language | Default evaluated questions |
|---|---|---:|
| HotpotQA | English | 100 |
| MuSiQue | English | 200 |
| ARCD | Arabic | 100 |

The configured sampling seed is `42`.

Dataset loading and preprocessing are defined in `data.py`.
The retrieval corpus is constructed from the passages associated
with the sampled questions, rather than a full Wikipedia collection.

## Project files

Keep the Python modules and supplied CSV files in the project root:

```text
.
├── README.md
├── app.py
├── config.py
├── data.py
├── indexing.py
├── llm.py
├── metrics.py
├── run_eval.py
├── strategies.py
├── results_hotpotqa.csv
├── results_musique.csv
├── results_arcd.csv
├── results.csv
└── summary.csv
```

- `config.py`: sampling, chunking, model, and retrieval settings
- `data.py`: dataset loading and corpus/chunk construction
- `indexing.py`: dense, lexical, and graph indexing
- `llm.py`: local Ollama calls and JSON parsing
- `strategies.py`: retrieval implementations and answer prompts
- `metrics.py`: answer, retrieval, and LLM-judge evaluation
- `run_eval.py`: command-line benchmark runner and CSV export
- `app.py`: experimental Streamlit interface
- `results_*.csv`: per-question results for individual datasets
- `results.csv`: combined per-question results
- `summary.csv`: mean scores grouped by dataset and method

## Reported environment

The project author's supplied environment snapshot contains:

| Component | Reported version |
|---|---|
| Python | 3.13.12 |
| Ollama application/CLI | 0.40.9 |
| Ollama Python client | 0.6.3 |
| PyTorch | 2.11.0+cu126 |
| sentence-transformers | 6.2.0 |
| transformers | 5.22.1 |
| datasets | 5.1.0 |
| NumPy | 2.4.3 |
| pandas | 3.0.1 |
| spaCy | 3.8.16 |
| English spaCy model | en_core_web_sm 3.8.0 |
| NetworkX | 3.6.1 |
| rank-bm25 | 0.2.2 |

This records the supplied environment; it is not a claim that a clean
installation of every listed version has been independently tested.

The Ollama application and the `ollama` Python package are separate
components and have different version numbers.

## Installation

Download or clone this repository and open a terminal in its directory.

### Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Install dependencies

Install a PyTorch build appropriate for your operating system and GPU.
The author's reported build is `torch==2.11.0+cu126`.

Install the remaining core dependencies:

```powershell
python -m pip install datasets sentence-transformers transformers numpy pandas spacy networkx rank-bm25 ollama tqdm
```

Install the English entity-recognition model:

```powershell
python -m pip install [https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl](https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl)
```

For the optional Streamlit interface:

```powershell
python -m pip install streamlit
```

These commands install dependencies but do not pin the complete
reported environment. Preserve the original environment separately:

```powershell
python -m pip freeze > requirements-environment.txt
```

### Set up Ollama

Install Ollama and download the local generator:

```powershell
ollama pull qwen3:8b
```

Ensure that the Ollama service is running. If it is not already running:

```powershell
ollama serve
```

Run the benchmark from a separate terminal.

Internet access is needed initially to obtain datasets, embedding
models, the spaCy model, and the Ollama model.
Generation requests in this implementation use local Ollama.

## Configuration

The default settings in `config.py` are:

| Setting | Value |
|---|---|
| Sampling seed | 42 |
| Final retrieval limit | 5 chunks |
| Chunk size | 200 words |
| Chunk overlap | 40 words |
| Default embedding model | `all-MiniLM-L6-v2` |
| ARCD embedding model | `paraphrase-multilingual-MiniLM-L12-v2` |
| Generator | `qwen3:8b` |
| Requested context window | 6144 tokens |
| Default request temperature | 0.0 |
| Maximum agent steps | 4 |
| Candidates per agent step | 6 |

The supplied Ollama model inspection reports approximately
8.2 billion parameters and `Q4_K_M` quantization.

The LLM wrapper explicitly requests `think=False`.
HyDE uses temperature `0.4` for its intermediate passage.

The application's request settings take precedence over the model
defaults shown by `ollama show` for the explicitly specified options.
For example, the benchmark requests a 6144-token context even though
the inspected model advertises a larger supported context.

## Run the benchmark

**Back up the supplied result CSVs before running an evaluation.**
A new run writes result files with the same names.

### All datasets and methods

```powershell
python run_eval.py
```

### One dataset

```powershell
python run_eval.py --datasets hotpotqa
```

```powershell
python run_eval.py --datasets musique
```

```powershell
python run_eval.py --datasets arcd
```

### Selected methods

For example, compare Naive, Hybrid, and Agentic v2 on MuSiQue:

```powershell
python run_eval.py --datasets musique --methods Naive Hybrid Agentic
```

### Smaller trial run

```powershell
python run_eval.py --datasets hotpotqa --methods Naive --scale 0.1 --no-judge
```

This requests 10 HotpotQA questions using the default configuration.
It is a smoke test, not a reproduction of the full benchmark.

Changing the sample size also changes the pooled retrieval corpus,
so scaled runs are not directly comparable to the reported full runs.

### Disable LLM-judge metrics

```powershell
python run_eval.py --no-judge
```

### Judge every fifth question

```powershell
python run_eval.py --judge-every 5
```

Unjudged rows omit the judge scores.
Interpret aggregate judge scores using the actual evaluated coverage.

### Resume completed methods

```powershell
python run_eval.py --resume
```

The current resume logic skips a method if that method already appears
in the dataset result file. It does not check that all questions for
that method were completed.

Do not use `--resume` to recover an interrupted, partially completed
method without first checking the CSV contents.

## Evaluation outputs

The runner writes:

```text
results_hotpotqa.csv
results_musique.csv
results_arcd.csv
results.csv
summary.csv
```

A dataset result contains the following fields:

```text
dataset
method
qid
question
gold
pred
f1
em
recall@5
precision@5
mrr
retrieval_time
generation_time
total_time
llm_calls
context_precision
faithfulness
answer_relevancy
```

Judge columns are included when judge evaluation is enabled.

The `qid` field is the question's position in the sampled dataset,
not necessarily the original dataset's native identifier.

The aggregate summary averages numerical metrics by dataset and method
and rounds the exported means to four decimal places.

A run restricted to selected datasets or methods produces combined
outputs for that invocation, not necessarily the entire benchmark.

## Metrics

- **F1:** token-overlap answer score
- **EM:** normalized exact-match answer score
- **Recall@5:** coverage of gold source titles within the retrieved context
- **Precision@5:** source-title coverage divided by the retrieval cutoff,
  as implemented in `metrics.py`
- **MRR:** reciprocal rank of the first matching gold source title
- **Retrieval time:** time spent in the retrieval strategy
- **Generation time:** time spent generating the final answer
- **Total time:** retrieval time plus final-answer generation time
- **LLM calls:** retrieval and final-answer calls recorded before judging
- **Context precision, faithfulness, and answer relevancy:** custom
  prompt-based LLM-judge scores

Consult `metrics.py` for the exact normalization, formulas, prompts,
and parsing behaviour. The judge scores should not be interpreted
as human annotations or automatically equated with another evaluation
library's implementation.

Judge execution occurs after the recorded retrieval/generation timing
and model-call count for each question.

## Inspect the saved results

The supplied CSVs can be inspected without rerunning model inference:

```powershell
python -c "import pandas as pd; print(pd.read_csv('summary.csv').to_string(index=False))"
```

For per-question inspection:

```powershell
python -c "import pandas as pd; print(pd.read_csv('results_hotpotqa.csv').head().to_string(index=False))"
```

## Reproducibility notes

- Preserve the final code and original CSVs together.
- Record the exact command and command-line options used for each run.
- Preserve dependency versions and model identifiers.
- Dataset sampling uses seed 42, but no explicit Ollama generation seed
  is supplied by the current LLM wrapper.
- A zero temperature does not constitute a guarantee of identical
  generated answers across environments.
- Timing depends on hardware, model loading, caching, and runtime load.
- The runner does not contain an explicit LLM warm-up call.
- The CSVs contain answers, scores, timings, and call counts, but do not
  preserve complete retrieved contexts or internal routing traces.
- Rerunning inference may produce different answers, judge scores,
  and latency measurements.

The saved CSVs document the original measured outcomes; rerunning the
code is a separate attempt to repeat the experiment.

## Dataset and model attribution

HotpotQA, MuSiQue, ARCD, Qwen3, the embedding models, and the spaCy model
are third-party resources. Cite their original publications or model
documentation when using this work and follow their applicable terms.

