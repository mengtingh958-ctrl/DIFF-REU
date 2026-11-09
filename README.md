# Diff-REU Public Pipeline

This archive is a **public evaluation interface** for Diff-REU. It shows how each experimental fold is run. The sample selection implementation is not included. Therefore, this code alone cannot reproduce the results reported in the tables of the paper.

## What is included

`diffreu_pipeline.py` exposes three functions:

1. `train_diffusion(X_train, y_train, feature_info, num_classes)`  
   Fit a tabular diffusion model on **majority class samples only**.
2. `undersample(diffusion, X_train, y_train) -> X_bal, y_bal`  
   Black-box call that returns a balanced training set.
3. `train_classifier(X_bal, y_bal)`  
   Train a Random Forest (`n_estimators=250`) on the balanced data.

`run_one_fold(...)` chains the three steps and reports test ROC-AUC, Gini, G-mean, F1, and TPRGap.

Label convention: **1 = minority**, **0 = majority**, matching the paper experiments.

## What is not included

The sample selection module is withheld. Without a local `diffreu_core.py`, `undersample()` raises an error by design, and `train_diffusion()` returns a handle only.

## Requirements

Python 3.8 or newer.

This public file only imports:

```text
numpy>=1.20
scikit-learn>=1.0
```

The paper experiments additionally use:

```text
torch>=1.12
pandas>=1.3
scipy>=1.7
imbalanced-learn>=0.9
```

A CUDA build of PyTorch is recommended but not required. `pandas` and `scipy` are used in data loading and ranking statistics; `imbalanced-learn` (`imblearn`) is used for baseline undersamplers such as Tomek Links and ENN.

## Usage

```python
from diffreu_pipeline import run_one_fold, train_diffusion, undersample, train_classifier

# Full fold (requires diffreu_core.py in the same directory)
out = run_one_fold(X_train, y_train, X_test, y_test, feature_info, num_classes)
print(out["auc"], out["gini"], out["gmean"], out["f1"], out["tprgap"])

# Or step by step
diffusion = train_diffusion(X_train, y_train, feature_info, num_classes)
X_bal, y_bal = undersample(diffusion, X_train, y_train)
clf = train_classifier(X_bal, y_bal)
y_score = clf.predict_proba(X_test)[:, 1]
```

Dry-run without the private module:

```bash
python diffreu_pipeline.py
```

This command builds a diffusion handle on a synthetic imbalanced dataset and then stops at `undersample()`, which is expected.

If a private backend is present, it should be a file named `diffreu_core.py` in the same directory, defining `train_majority_diffusion(...)` and `select_majority(diffusion, X, y) -> (X_bal, y_bal)`. No change to `diffreu_pipeline.py` is required.

## Data format

| object | meaning |
|---|---|
| `X_*` | `float` array, shape `(n_samples, n_features)` |
| `y_*` | integer labels, shape `(n_samples,)` |
| `feature_info` | dict; at least `{"num_numerical": int}` |
| `num_classes` | sequence of categorical cardinalities; empty or `None` if all features are numeric |

Data-loading and train/test-split scripts are not included in this archive.

## Supplementary PDF

If `supplementary.pdf` is included, it contains extra experimental results that did not fit in the main paper (CD diagrams, sensitivity, ablation, and extra metrics). It does not include the sample selection implementation.
