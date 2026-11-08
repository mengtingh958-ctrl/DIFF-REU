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

Python 3.8 or newer.

This public file only imports:

```text
numpy>=1.20
scikit-learn>=1.0
```

To train the tabular diffusion backbone used in the paper (TabDDPM-style, majority class only), also install:

```text
torch>=1.12
pandas>=1.3
scipy>=1.7
imbalanced-learn>=0.9
```

A CUDA build of PyTorch is recommended but not required. `pandas` and `scipy` are used in data loading and ranking statistics; `imbalanced-learn` (`imblearn`) is used for baseline undersamplers such as Tomek Links and ENN. They are not imported by `diffreu_pipeline.py` as released here.

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

## Supplementary Material

Export the supplementary material as a single PDF named `supplementary.pdf` and place it alongside the README and pipeline script:

```text
README.md
diffreu_pipeline.py
supplementary.pdf
```

### PDF Contents

The supplementary PDF may include:

- Additional tables and figures, including CD diagrams, sensitivity analyses, ablation results, and evaluation metrics omitted from the main paper.
- evaluation protocol details.
- Assumptions, propositions, and detailed proofs for the convergence analysis.

Keep installation instructions, dependencies, and execution commands in this README.

If the code archive exposes only the pipeline interface, ensure that the methodological details in the PDF match the intended scope of disclosure.

### Preparation and Packaging

1. Compile the supplementary LaTeX source into a single PDF.
2. Rename the PDF to `supplementary.pdf`.
3. Place the three files listed above directly at the root of the ZIP archive, without an enclosing folder.

