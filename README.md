# Evaluation Metrics for NLP — Classification, Sequence Labeling, Machine Translation

**Master Data & IA · ML Evaluation · INPT · 2026 — Kaoutar Chakroun**

This project explains and demonstrates the evaluation metrics of three NLP tasks.
The presentation teaches the concepts; the notebook applies every metric to **real datasets** with **19 charts**.

## Repository content

| File | Description |
|---|---|
| `nlp_evaluation_metrics.ipynb` | Main notebook: the metrics applied to 4 real datasets, 19 charts (already executed) |
| `classification_metrics.py` | Classification metrics: accuracy, precision, recall, F1, ROC-AUC, micro/macro/weighted F1, accuracy trap |
| `sequence_labeling_metrics.py` | Sequence labeling (NER, BIO): token accuracy, entity-level precision/recall/F1, error types |
| `translation_metrics.py` | Machine translation: corpus BLEU, chrF, TER, sentence BLEU/chrF/METEOR |
| `requirements.txt` | Python dependencies |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook nlp_evaluation_metrics.ipynb
```

On **Google Colab**: open the notebook and uncomment the first cell (`!pip install ...`).
All datasets are downloaded automatically through `nltk.download` (no manual download). Full run: about 3 minutes on a CPU.

Each script also runs on its own with a small example:

```bash
python classification_metrics.py
python sequence_labeling_metrics.py
python translation_metrics.py
```

## Metric scripts

| Script | Main functions |
|---|---|
| `classification_metrics.py` | `evaluate_classification(y_true, y_pred, y_proba=None, average="binary")` → accuracy, precision, recall, F1 (+ ROC-AUC) · `f1_averages(y_true, y_pred)` → micro / macro / weighted F1 · `accuracy_trap()` → 95 % accuracy with F1 = 0 |
| `sequence_labeling_metrics.py` | `get_entities(tags)` → BIO tags to `(type, start, end)` entities · `token_accuracy(y_true, y_pred)` · `entity_scores(y_true, y_pred)` → entity-level P / R / F1 (CoNLL, cross-checked with seqeval) · `error_types(y_true, y_pred)` → boundary / type / missed / spurious |
| `translation_metrics.py` | `evaluate_translation(hypotheses, references)` → corpus BLEU, chrF, TER, n-gram precisions, brevity penalty · `sentence_scores(hypothesis, reference)` → sentence BLEU, chrF, METEOR |

## Datasets and models

| Part | Task | Dataset | Model |
|---|---|---|---|
| 1 | Binary classification (sentiment) | NLTK **Movie Reviews**: 2,000 reviews, 50 % positive / 50 % negative | TF-IDF (1–2-grams) + Logistic Regression |
| 2 | Multi-class classification (topics) | **Reuters-21578**: single-label articles of the 8 most frequent topics (5,557 train / 2,205 test) | TF-IDF + Logistic Regression |
| 3 | Named Entity Recognition | **CoNLL-2002** Spanish: 8,323 train / 1,517 test sentences, BIO tags for PER, ORG, LOC, MISC | CRF (`sklearn-crfsuite`) with word, suffix, case and POS features |
| 4 | Machine translation (French → English) | **ComTrans** (Europarl) French–English sentence pairs; 400 test sentences | IBM Model 1 (`nltk.translate`) trained on 500 / 3,000 / 12,000 pairs + a copy-source baseline |

## Notebook walkthrough and documentation

### Part 1 — Binary classification
| Step | Code | What it shows |
|---|---|---|
| Data exploration | `value_counts`, histogram | Balanced classes, review length |
| Model | `make_pipeline(TfidfVectorizer, LogisticRegression)` | 75/25 stratified split |
| 1.1 Metrics | `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix` | Accuracy 0.864, F1 0.864, ROC-AUC 0.943 |
| 1.2 Curves | `roc_curve`, `precision_recall_curve` | ROC curve and precision–recall curve |
| 1.3 Threshold | loop over thresholds | Precision ↑ / recall ↓ trade-off, best F1 threshold |
| 1.4 Imbalance | positives subsampled from 50 % to 5 %, `DummyClassifier` | Dummy model reaches 95 % accuracy with F1 = 0 |

### Part 2 — Multi-class classification
| Step | Code | What it shows |
|---|---|---|
| Data | `nltk.corpus.reuters` | Very imbalanced topics (earn ≫ sugar) |
| Report | `classification_report` | Per-class precision / recall / F1 |
| Averages | `f1_score(average="micro" / "macro" / "weighted")` | Micro-F1 0.953 vs macro-F1 0.872 |
| Errors | normalized `confusion_matrix` + heatmap | Small classes (ship, interest) confused with big ones |

### Part 3 — Named Entity Recognition
| Step | Code | What it shows |
|---|---|---|
| Data | `conll2002.iob_sents` | BIO tags; 88 % of tokens are `O` |
| Features | `w2f(sentence, i)` | Word, suffixes, capitalization, POS and neighbouring words |
| Model | `train_crf(X, y)` | CRF trained with L-BFGS |
| 3.1 Evaluation | `accuracy_score` on tokens vs `seqeval` | Token accuracy 0.971 vs entity F1 0.788 |
| Per type | `seqeval.classification_report` | PER best, MISC worst |
| 3.2 Error analysis | `get_entities` + overlap rules | Boundary / type / missed / spurious errors, type-confusion heatmap |
| 3.3 Learning curve | CRF trained on 250 → 8,323 sentences | Entity F1 rises from 0.59 to 0.79, token accuracy stays ≈ 0.95 |

### Part 4 — Machine translation
| Step | Code | What it shows |
|---|---|---|
| Data | `comtrans.aligned_sents("alignment-en-fr.txt")` | French source, English reference |
| Models | `train_ibm(n)` (IBM Model 1 + word-by-word decoding) | Four systems of increasing quality |
| 4.1 Metrics | `sacrebleu.corpus_bleu / corpus_chrf / corpus_ter`, `meteor_score` | All metrics rank the systems the same way |
| 4.2 Inside BLEU | `bleu.precisions`, `bleu.bp` | n-gram precisions collapse from 1-gram to 4-gram |
| 4.3 Sentence level | `sentence_bleu`, `sentence_chrf` | Sentence BLEU is low and noisy; correlation with chrF |
| 4.4 Significance | bootstrap resampling (300×) | 95 % confidence intervals of BLEU |
| 4.5 Paraphrases | synonym / paraphrase / word-order candidates | BLEU drops the most when wording changes |

## Main results

| Task | Key numbers | Metric to report |
|---|---|---|
| Sentiment | Accuracy 0.864 · F1 0.864 · ROC-AUC 0.943 | F1, ROC-AUC, PR curve |
| Reuters topics | Micro-F1 0.953 · Macro-F1 0.872 | Macro-F1 + per-class report |
| NER | Token accuracy 0.971 · Entity F1 0.788 | Entity-level F1 (seqeval) |
| Translation | BLEU 1.4 → 13.4 · chrF 27.5 → 44.1 (baseline → IBM-1 12k) | BLEU + chrF (+ COMET) with confidence intervals |
