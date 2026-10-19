# Urdu Text Analyzer

A small, reproducible baseline for three-way sentiment classification and exploratory topic discovery on Urdu-script text. It is intended for local experimentation, not high-stakes decisions.

## Problem and labels

Given one Urdu text, predict a polarity label and return class probability scores:

- `positive`: favorable or positive polarity (`P` in the reference corpus)
- `negative`: unfavorable or negative polarity (`N`)
- `neutral`: objective, mixed, or no clear polarity (`O`)

The source corpus is manually labeled tweets and includes short or ambiguous items. These labels are inherited rather than re-annotated by this project. The model's `confidence` is the largest logistic-regression probability, not a guarantee of correctness or a calibrated probability.

## Dataset and licensing

The reference dataset is the [Urdu Sentiment Corpus v1](https://github.com/MuhammadYaseenKhan/Urdu-Sentiment-Corpus), provided as `urdu-sentiment-corpus-v1.tsv` with `Tweet` and `Class` columns and `P`/`N`/`O` labels. The repository points to the associated papers by Khan and Nizami (2020) and Khan et al. (2017); please cite them when using the corpus.

**License status:** the upstream repository does not contain a dataset license or an explicit license grant (checked 2026-10-19). Do not assume this dataset may be redistributed or used commercially. This project intentionally does not bundle it or download it automatically. Obtain permission or confirm applicable terms with the data authors before use. Once authorized, place the TSV at `data/raw/urdu-sentiment-corpus-v1.tsv` or pass its path to `prepare`.

The project code is MIT licensed; that license does not apply to external datasets.

## Setup

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Prepare, train, and evaluate

After obtaining an authorized TSV:

```bash
urdu-analyzer prepare data/raw/urdu-sentiment-corpus-v1.tsv --output data/splits --seed 42
urdu-analyzer train --splits data/splits --seed 42
```

The first command accepts either `Tweet`/`Class` or `text`/`label` TSV headers and writes UTF-8 `train.jsonl`, `validation.jsonl`, and `test.jsonl`. The default target proportions are 80/10/10; actual sizes can differ because repeated normalized texts stay together and small classes cannot be subdivided exactly. Identical normalized text with conflicting labels is rejected. Seed is fixed by default and recorded as a command option. Keep split files private if source data terms require it.

Training uses only the training split. Validation is kept untouched for later model selection (this baseline has no tuned hyperparameters). The test split is used once for the final report. Run evaluation only after preparing the splits. Metrics are written to `reports/evaluation.json`: accuracy, per-class precision/recall/F1/support, macro and weighted summaries, and a confusion matrix with its label order. No benchmark score is claimed in this repository; scores must come from a run on an authorized dataset.

## Prediction

```bash
urdu-analyzer predict "یہ فلم بہت اچھی ہے، مجھے پسند آئی۔"
```

Output is JSON with `sentiment`, top-class `confidence`, probabilities in `scores`, and up to three matching topic suggestions. It requires a model produced by the training command. Before training, the human-labeled examples in [`examples/demo.json`](examples/demo.json) demonstrate expected input/output semantics only; they are not model predictions or evaluation data.

## Model and preprocessing

The sentiment baseline is scikit-learn logistic regression over character TF-IDF n-grams of lengths 2–5. This avoids relying on language-specific tokenizers and tends to tolerate spelling variation, while remaining an interpretable baseline. Input is normalized with Unicode NFC; URLs and `@mentions` become stable tokens, tatweel and control/format characters are removed, and whitespace is collapsed. Urdu script, punctuation, diacritics, and letter variants are preserved. There is no transliteration, stemming, stop-word removal, or letter folding. The same normalization is applied at training and inference.

Topics use non-negative matrix factorization (NMF) over character TF-IDF on training text. Topic terms are fragments rather than guaranteed words, and this is unsupervised topic discovery, not a topic classifier. The analyzer reports topic terms; candidate topics are suggested for new text by exact substring overlap against those fragments. Inspect and name topics manually. Evaluate topic usefulness qualitatively by reviewing representative documents and checking whether top terms form coherent themes; no quantitative topic score is claimed.

## Known limitations

- The reference corpus is small, tweet-focused, and may be politically or temporally dated; it does not represent all Urdu writing, dialects, or domains.
- Tweet polarity labels can be ambiguous, and neutral may include mixed or weakly subjective text.
- Character n-gram probabilities are not calibrated. Sarcasm, negation scope, code-switching, spelling noise, and context can lead to errors.
- NMF topics depend on corpus and seed, and character fragments need human interpretation.
- The repository does not include dataset data or a pretrained model, so prediction requires training on authorized data first.

## Repository layout

```text
src/urdu_text_analyzer/  preprocessing, data, model and CLI
data/raw/                local source data (ignored by git)
data/splits/             generated split files (ignored by git)
models/                  trained model and discovered topics (ignored by git)
reports/                 generated evaluation report (ignored by git)
examples/                labeled interface examples
```

## Citation

Khan, Muhammad Yaseen, and Muhammad Suffian Nizami. “Urdu Sentiment Corpus (v1.0): Linguistic Exploration and Visualization of Labeled Dataset for Urdu Sentiment Analysis.” ICISCT, 2020. See the upstream repository for this and the 2017 baseline citation.
