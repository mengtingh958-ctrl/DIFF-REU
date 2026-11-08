# Diff-REU Public Pipeline

This folder is a **public evaluation interface** for Diff-REU. It shows how each experimental fold is run, without releasing the sample-selection implementation.

## What is included

`diffreu_pipeline.py` exposes three functions:

1. `train_diffusion(X_train, y_train, feature_info, num_classes)`  
   Fit a tabular diffusion model on **majority-class samples only**.
2. `undersample(diffusion, X_train, y_train) -> X_bal, y_bal`  
   Black-box call that returns a balanced training set. The selector itself is not in this release.
3. `train_classifier(X_bal, y_bal)`  
   Train a Random Forest (`n_estimators=250`) on the balanced data.

`run_one_fold(...)` chains the three steps and reports test ROC-AUC.

Label convention: **1 = minority**, **0 = majority**, matching the paper experiments.

## What is not included

The following are withheld and will be released after acceptance:

- two-stage majority-class selection
- scoring rules and their combination weights
- noise filtering
- the repeated undersampling / ensemble procedure

Do **not** add those details to this folder if the goal is to avoid disclosing the method.

## Requirements

```text
numpy
scikit-learn
```

Optional, only needed when the private backend is present:

```text
torch
```

## Usage

```python
from diffreu_pipeline import run_one_fold, train_diffusion, undersample, train_classifier

# Full fold (requires diffreu_core.py next to this file)
out = run_one_fold(X_train, y_train, X_test, y_test, feature_info, num_classes)
print(out["auc"])

# Or step by step
diffusion = train_diffusion(X_train, y_train, feature_info, num_classes)
X_bal, y_bal = undersample(diffusion, X_train, y_train)   # needs private module
clf = train_classifier(X_bal, y_bal)
y_score = clf.predict_proba(X_test)[:, 1]
```

Dry-run the public file without the private module:

```bash
python diffreu_pipeline.py
```

This trains the public diffusion handle on a synthetic imbalanced set and then stops at `undersample()`, which is expected.

## Connecting the private module later

Place `diffreu_core.py` in the same directory with:

```python
def train_majority_diffusion(X_majority, feature_info=None, num_classes=None,
                             num_steps=6000, batch_size=128, seed=42):
    ...
    return model

def select_majority(diffusion, X_train, y_train):
    ...
    return X_bal, y_bal
```

No change to `diffreu_pipeline.py` is required.

## Data format

| object | meaning |
|---|---|
| `X_*` | `float` array, shape `(n_samples, n_features)` |
| `y_*` | integer labels, shape `(n_samples,)` |
| `feature_info` | dict; at least `{"num_numerical": int}` |
| `num_classes` | sequence of categorical cardinalities; empty or `None` if all features are numeric |

Loaders and stratified CV used in the paper are separate from this interface and may be shared independently (dataset I/O only).
# DIFF-REU
