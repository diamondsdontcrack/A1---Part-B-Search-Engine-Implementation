# ISYS3476 (Postgraduate) - Assignment 1 Part B

This is the Search Engine Implementation starter for Managing Semi-structured and Unstructured Data. It follows the required module layout and command-line interfaces used by the assignment commands and sanity checker.

Part B builds on Part A. Your submission must meaningfully reuse or adapt your own Part A work in at least one relevant implementation. Simply retaining unused utility files does not meet this requirement. You may copy your completed Part A `utils/*.py` files into this repository because the utility function contracts are kept consistent.
Part B includes a system dev collection under `data/dev/` and a dedicated Task 2 ranking collection under `data/dev_ranking/`. Task 2 uses fixed tokenization so its Pearson score measures ranking rather than Part A preprocessing. You may use Part A preprocessing as part of a Task 3 system optimisation.

## Submission

Submit one ZIP of your completed assignment on Canvas. Package your code, runs, `WRITEUP.pdf`, and `DEMONSTRATION.mp4` or `DEMONSTRATION.mov` together in this ZIP. Place the recording at the repository root. External video links are not marked.

Name the submitted ZIP as `YourName_s123456.zip` (e.g., `ZhuangLi_s123456.zip`).

Only the ZIP submitted on Canvas is marked.

Do not include virtual environments such as `.venv/`, `env/`, `venv/`, or `.conda/`.

## Generative AI Condition

This assessment uses **Condition 2: AI for surface polish**. Generative AI may
only improve grammar, spelling, clarity, formatting, and language in text you
have already written for `WRITEUP.pdf`. It must not generate or change technical
content, code, analysis, report structure, or demonstration content. If you use
Generative AI, add the declaration specified for `WRITEUP.pdf`. No declaration
is required if you do not use Generative AI. See the Main Specification and the
Canvas [Generative AI Use in Assessment](https://rmit.instructure.com/courses/157398/pages/generative-ai-use-in-assessment)
page before using any Generative AI tool. The Main Specification is authoritative.

## Setup

The VS Code steps below provide a guided setup. If you are comfortable with Python environments or the terminal, use your preferred setup and skip to the quick checks. In either case, use Python 3.10 to 3.14, install `requirements.txt` and the required NLTK resources, and follow the permitted-import list in the specification.

For additional help, see [Python Setup and Troubleshooting](https://rmit.instructure.com/courses/157398/pages/python-setup-and-troubleshooting?module_item_id=8766153).

For guided setup in VS Code:

1. Download the starter ZIP from Canvas.
2. Extract the ZIP first. On Windows, right-click it and select **Extract All**. On macOS, double-click it. Do not work inside the ZIP.
3. Open VS Code, select **File > Open Folder**, and choose the extracted `ir-ass1-partB-startcode-PG` folder itself.
4. If VS Code displays **Restricted Mode**, select **Manage** and trust the extracted course folder. Confirm that `README.md`, `requirements.txt`, `index/`, and `query_processing/` are visible in the VS Code file list.
5. Press **Ctrl+Shift+P** on Windows or **Cmd+Shift+P** on macOS. Type `Python: Create Environment` and select it from the results. On the next screen, do not select **Quick Create**. Select **venv**, the second option. If the command does not appear, install or update the Microsoft Python and Python Environments extensions in VS Code.
6. Select an installed Python version from 3.10 to 3.14. If no supported version is listed, install one from [python.org](https://www.python.org/downloads/) and reopen VS Code.
7. Select **Install project dependencies**. On the next screen, keep `requirements.txt` selected and click **OK**. VS Code will create a local `.venv`, install the packages, and select the new environment for this folder.
8. When environment creation finishes, select **Terminal > New Terminal**. The terminal prompt should begin with `(.venv)`, which confirms that the new environment is active. Then run:

```bash
python -m nltk.downloader punkt punkt_tab stopwords
```

The self-checker also attempts to download any missing NLTK tokenizer or stopword data.

On Windows, if PowerShell reports that script execution is disabled, select **Terminal > Select Default Profile > Command Prompt**, then open a new terminal. You do not need to change the system execution policy.

For help importing functions between project folders or resolving module errors, see [Python Project Imports and Running Modules](https://rmit.instructure.com/courses/157398/pages/python-project-imports-and-running-modules).

## Quick Checks

Run the following commands from the same VS Code terminal. Its current folder must be the extracted starter folder containing this `README.md`.

```bash
python -m test_sanity.check_submission

# In the starter, default initially points to keyword_match.
python -m metrics.eval_pear keyword_match
python -m metrics.eval_pear default

python -m system.search_system data/dev/queries.json data/dev/documents.jsonl runs/run_no_optimisation.json no_optimisation

# In the starter, default initially matches no_optimisation.
python -m system.search_system data/dev/queries.json data/dev/documents.jsonl runs/run_default.json default

python -m metrics.eval_map
```

The sanity checker verifies interfaces and small examples. It does not measure ranking quality. Use `metrics.eval_pear` for Task 2 dev guidance and `metrics.eval_map` for Task 3 dev guidance.

## Recorded CLI Demonstration

Submit one continuous, unedited screen recording of no more than 3 minutes. From the
repository root, type and run the self-checker, then type and run either
`python -m metrics.eval_pear default` or `python -m metrics.eval_map`. From
the Pearson or MAP output, choose one visible result. Open one code block in
your submission that is relevant to that result, and explain how its underlying
information retrieval principle contributed to the result. See the Main
Specification and rubric for the full requirements.

Commands and file navigation must occur live. Visible recording edits, such as
cuts, splicing, speed changes, or replaced audio, result in 0 for this component.

## Project Folders

```text
index/               index construction and access
query_processing/    structured query processors
ranking/             Task 2 ranker
system/              Task 3 retrieval pipeline
metrics/             development evaluators
utils/               reusable Part A utilities
data/                provided development data
runs/                generated run files
test_sanity/         public self-checker
```

See the Main Specification for the exact required submission layout.

## Provided vs To Implement

Provided:

- `index/io.py` serialization helpers using `pathlib`, `gzip`, and `pickle`.
- The unified term-index construction, package schema, and access helpers for Task 1.
- Selected workshop-derived Task 1 query-processing scaffolding. It covers only the
  subset implemented in the workshop.
- A minimal keyword-match ranker so the end-to-end system can run.
- The Week 03 cosine-similarity helper. You still construct representations,
  score candidates, and produce the complete ranking.
- Dev evaluation helpers for Pearson and MAP.

In the Week 05 handout, Assignment Part B Task 3 refers to the ranking work
assessed as Task 2 in this release.

To implement:

- Task 1: complete the wildcard and proximity indexes and implement the Boolean, wildcard, and proximity processors.
- Task 2: improve `rank_documents(...)` while returning all candidate docs.
- Task 3: implement one non-ranking optimisation and write the required run files. In `system/search_system.py`, `no_optimisation` is the baseline system, and `default` is your submitted final system. In the starter, both initially do the same thing.

## Important Contracts

- Do not change required function names, argument order, or return types.
- `create_all_indexes(...)` receives already-cleaned tokens. Use tokens as-is.
- Task 1 writes one index package containing `__META__`, `unified`, `wildcard`, and `proximity`.
- Dev helpers and grading tests may rebuild an index at the same path for a different corpus. Do not skip rebuilding only because the file already exists; invalidate any in-memory cache when the index file changes.
- `find_wildcard_matches(ngram, index_path)` is a character n-gram lookup, not a full wildcard-pattern processor.
- `get_all_doc_ids(...)` returns all indexed document IDs in ascending order.
- Task 1 queries are well formed. Boolean operands are unigrams.
- Wildcard tests use prefix, suffix, and internal patterns, such as `climat*`, `*ing`, and `learn*ing`. The `*` may match the empty sequence.
- Task 3 MAP uses natural-language and Boolean queries. Wildcard and proximity correctness are assessed in Task 1.
- If you preprocess structured queries for Task 3, preserve Boolean operators,
  parentheses, wildcard syntax, and `NEAR/k` syntax before query processing.
- `rank_documents(...)` must return a complete ranking: `ranked_doc_ids` is a permutation of the input `doc_ids`, with one score per doc.
- `system/search_system.py` writes a JSON list with `qid`, `doc_ids`, and `scores`, keeping at most 10 documents per query.
