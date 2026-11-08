# NLP Pipeline Introduction Workshop

Course: **PROG8245-26F-Sec1 — Machine Learning Programming**
Lab: **NLP Pipeline Introduction Workshop**

## Team Members - Group 3

| Name | Student ID |
|------|------------|
| Juan Camilo Chirivi  | 9115141 |
| Zeynep Ozdemir       | 9045142 |
| Senay Teweldebrhan   | 9120588 |
| Ricardo Mohammed     | 7500382 |

An end-to-end walkthrough of a classical NLP preprocessing pipeline on the
**20 Newsgroups** dataset, followed by an extractive question-answering demo
using **DistilBERT**, plus four hands-on challenge experiments that probe
preprocessing trade-offs and LLM tokenization/cost.

---

## What's in this repo

| File | Purpose |
|---|---|
| `NLP-Pipeline-Introduction.ipynb` | The main workshop notebook (fully executed) |
| `requirements.txt` | Pinned Python dependencies |
| `README.md` | This file |
| `.gitignore` | Excludes `.venv/`, `nltk_data/`, checkpoints, caches |

### Notebook contents at a glance

1. **Environment setup** — installs deps, downloads NLTK data into `./nltk_data/`
2. **Data loading & EDA** — loads 20 Newsgroups (~18.8k docs) and prints initial stats
3. **Preprocessing pipeline** — tokenization → normalization + stop-word removal → Porter stemming
4. **Vocabulary comparison** — before vs. after pipeline (~288K → ~112K unique tokens)
5. **BERT extractive QA demo** — Hugging Face `question-answering` pipeline (DistilBERT-SQuAD)
6. **Four challenges** — negation preservation, stemming vs. lemmatization, Bag-of-Words, LLM token-cost reflection
7. **Conclusion** — summary of trade-offs and where the pipeline goes next

---

## Requirements

- Python **3.10+** (tested on 3.11 / 3.12)
- ~1.5 GB disk (model download + env)
- No GPU required; runs on CPU

> **Important:** this lab pins `transformers<5.0.0`. Version 5 of the library
> **removed** the `question-answering` pipeline, which would break the BERT demo.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/nlp-pipeline.git
cd nlp-pipeline
```

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (cmd)**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Register the venv as a Jupyter kernel

```bash
python -m ipykernel install --user --name nlp-pipeline --display-name "Python (nlp-pipeline)"
```

### 5. Launch Jupyter

```bash
jupyter notebook
```

Open `NLP-Pipeline-Introduction.ipynb`, select the `Python (nlp-pipeline)` kernel
(or whichever kernel points to your `.venv`), then run **Kernel → Restart & Run All**.

---

## First-run notes

- The **NLTK setup cell** downloads `punkt`, `punkt_tab`, `stopwords`, and
  `wordnet` into a local `./nltk_data/` directory. Run it once before any other
  cell. If you see `Resource punkt_tab not found`, just re-run that setup cell.
- The **BERT demo** downloads the `distilbert-base-cased-distilled-squad` model
  (~250 MB) on first run and caches it in `~/.cache/huggingface/`. Subsequent
  runs load from cache.
- Harmless warnings you can ignore:
  - `TqdmWarning: IProgress not found` → install `ipywidgets` to silence, or ignore.
  - Hugging Face **symlink warning on Windows** → harmless; caching still works.
- Full notebook runtime on CPU: **~3–5 minutes** after the first model download.

---

## Results summary

| Metric | Value |
|---|---|
| Documents loaded | ~18,846 |
| Initial unique tokens (vocab) | ~288,723 |
| Final unique tokens after pipeline | ~112,264 |
| Vocabulary reduction | **~61.1%** |
| DistilBERT QA confidence (definition question) | ~0.71 |

---

## Key takeaways

### 1. The pipeline is layered

`tokenize → lowercase + remove stop words → stem → compare vocabulary`.
Each stage depends on the previous one — order matters.

### 2. Every preprocessing choice is a trade-off

- **Negation handling** *(Challenge 1)* — removing `not`, `no`, `never` flips
  the meaning of sentences like *"this movie is **not** good."* The pipeline
  keeps a small negation-preservation list so sentiment signal survives.
- **Stemming vs. lemmatization** *(Challenge 2)* — stemming is fast but produces
  crude stems (`studies → studi`); lemmatization produces dictionary forms
  (`studies → study`) but is slower and usually needs POS tags. The right choice
  depends on whether humans must read the output or you need speed at scale.

### 3. From words to numbers *(Challenge 3)*

A Bag-of-Words view turns tokens into count vectors a classical ML model can
consume. It is simple and effective but discards word order, grammar, negation
scope, and semantic relations — which is exactly the gap modern contextual
embeddings (BERT, GPT) close.

### 4. Extractive vs. generative models

The BERT demo works because BERT is **bidirectional** and pre-trained on massive
text. It can *extract* an answer span with high confidence — but it cannot
reason, summarize, or understand humor. That is the boundary between an
extractive model and a generative LLM like ChatGPT / Gemini.

### 5. Tokenization and cost *(Challenge 4)*

- **Prompt tokens** are what you pay to *send*; **completion tokens** are what
  you pay to *receive* (usually 2–4× more expensive).
- The **only** preprocessing step that meaningfully reduces prompt cost is
  **stop-word removal** (with negation preservation) — about **30–40% fewer tokens**.
- **Stemming saves 0%** on token count — it changes word shapes, not counts.
- **Never send a Bag-of-Words dict to a chat model** — it destroys order and
  grammar while saving nothing.
- **Model choice** dominates cost more than prompt compression does.

### 6. The professional skill is judgment, not recipes

| Decision | Cost | Benefit |
|---|---|---|
| Lowercasing | Loses case signal | Smaller, cleaner vocab |
| Stop-word removal | May drop meaning (fix: keep negations) | ~35% fewer tokens |
| Stemming | Ugly stems | Fast vocab consolidation |
| Lemmatization | Slower, needs POS | Readable lemmas |
| Bag-of-Words | Loses order & context | Numeric features for ML |
| Pretrained transformers | Compute / cost | Contextual understanding, SOTA |

Good preprocessing isn't about deleting as much text as possible — it's about
**removing the right text, keeping the right signal, and being able to justify
the choice with measurements.**

---

## Repo structure

```
nlp-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── NLP-Pipeline-Introduction.ipynb   # fully executed
└── (nltk_data/ generated on first run — gitignored)
```

---

## Reproducing the full run

```bash
git clone https://github.com/<your-username>/nlp-pipeline.git
cd nlp-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook NLP-Pipeline-Introduction.ipynb
# Kernel → Restart & Run All
```

Expected outcome: all cells run without errors; the final cell prints a
vocabulary reduction around **61%** and the QA pipeline returns a confidence
score around **0.70**.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `KeyError: 'The task question-answering has been removed'` | You have `transformers>=5`. Run `pip install "transformers<5.0.0"` and restart the kernel. |
| `Resource punkt_tab not found` | Re-run the NLTK setup cell; it downloads `punkt_tab`. |
| `TqdmWarning: IProgress not found` | Harmless; `pip install ipywidgets` + restart kernel to silence. |
| Model download is slow | First run only; Hugging Face caches to `~/.cache/huggingface/`. |
| Long runtime on CPU | Expected (~3–5 min); first run includes ~250 MB model download. |

---

## License / attribution

- Dataset: 20 Newsgroups (scikit-learn built-in)
- QA model: `distilbert-base-cased-distilled-squad` (Hugging Face)
- Course materials: PROG8245, Conestoga College

For educational use only.