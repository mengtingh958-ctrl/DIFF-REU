# -*- coding: utf-8 -*-
"""
Public evaluation interface for Diff-REU.

This file is intentionally limited to the experiment pipeline:

    train diffusion on the majority class
        -> black-box undersampling
        -> train a classifier on the balanced set

The sample-selection implementation is NOT included.  Place a private
module named ``diffreu_core.py`` (with ``select_majority``) next to this
file when the full method is released.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ArrayLike = Union[np.ndarray, Any]

MINORITY_LABEL = 1
MAJORITY_LABEL = 0


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _majority_mask(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    if np.issubdtype(y.dtype, np.floating):
        y = y.astype(int)
    counts = {int(c): int((y == c).sum()) for c in np.unique(y)}
    maj = max(counts, key=counts.get)
    return y == maj, maj


@dataclass
class DiffusionHandle:
    """Trained majority-class diffusion backbone. Contains no selector state."""

    model: Any
    majority_class: int
    n_features: int
    feature_info: Optional[Dict[str, Any]]
    num_classes: Optional[Sequence[int]]


def train_diffusion(
    X_train: ArrayLike,
    y_train: ArrayLike,
    feature_info: Optional[Dict[str, Any]] = None,
    num_classes: Optional[Sequence[int]] = None,
    *,
    num_steps: int = 6000,
    batch_size: int = 128,
    seed: int = 42,
) -> DiffusionHandle:
    """
    Step 1. Train a tabular diffusion model on majority-class samples only.

    This function only fits the generative backbone.  It does not score
    samples, filter noise, or choose which majority instances to keep.

    If the private training backend is present, it is used; otherwise a
    lightweight handle is returned so that the public pipeline still type-checks.
    """
    X = _to_numpy(X_train)
    y = _to_numpy(y_train).reshape(-1)
    mask, maj = _majority_mask(y)
    X_maj = X[mask]
    print(
        "[Diff-REU] train_diffusion: majority=%d, n_maj=%d, n_min=%d, steps=%d"
        % (maj, int(mask.sum()), int((~mask).sum()), num_steps)
    )

    model: Any = None
    try:
        from diffreu_core import train_majority_diffusion  # optional private backend

        model = train_majority_diffusion(
            X_maj,
            feature_info=feature_info,
            num_classes=num_classes,
            num_steps=num_steps,
            batch_size=batch_size,
            seed=seed,
        )
    except ImportError:
        model = {"backend": "public-stub", "n_majority": int(len(X_maj)), "seed": seed}

    return DiffusionHandle(
        model=model,
        majority_class=int(maj),
        n_features=int(X.shape[1]),
        feature_info=feature_info,
        num_classes=num_classes,
    )


def undersample(
    diffusion: DiffusionHandle,
    X_train: ArrayLike,
    y_train: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 2. Black-box undersampling.

    Parameters
    ----------
    diffusion : DiffusionHandle
        Output of ``train_diffusion``.
    X_train, y_train
        Imbalanced training set. Label 1 is the minority class.

    Returns
    -------
    X_bal, y_bal : ndarray
        Balanced training set (selected majority samples + all minority samples).

    Notes
    -----
    Selection rules are implemented in ``diffreu_core.select_majority``,
    which is not part of this public release.
    """
    X = _to_numpy(X_train)
    y = _to_numpy(y_train).reshape(-1)
    try:
        from diffreu_core import select_majority
    except ImportError as exc:
        raise RuntimeError(
            "The Diff-REU sample-selection module is not included in this "
            "public interface. Expected a local file diffreu_core.py that "
            "defines select_majority(diffusion, X, y) -> (X_bal, y_bal). "
            "The full implementation will be released after acceptance."
        ) from exc

    X_bal, y_bal = select_majority(diffusion, X, y)
    X_bal = _to_numpy(X_bal)
    y_bal = _to_numpy(y_bal).reshape(-1)
    print(
        "[Diff-REU] undersample: n_in=%d -> n_out=%d (min=%d, maj=%d)"
        % (
            len(y),
            len(y_bal),
            int((y_bal == MINORITY_LABEL).sum()),
            int((y_bal == MAJORITY_LABEL).sum()),
        )
    )
    return X_bal, y_bal


def train_classifier(
    X_bal: ArrayLike,
    y_bal: ArrayLike,
    *,
    n_estimators: int = 250,
    seed: int = 42,
) -> RandomForestClassifier:
    """Step 3. Train a Random Forest on the balanced training set."""
    X = _to_numpy(X_bal)
    y = _to_numpy(y_bal).reshape(-1)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X, y)
    print("[Diff-REU] train_classifier: RF(%d trees) on %d samples" % (n_estimators, len(y)))
    return clf


def run_one_fold(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    y_test: ArrayLike,
    feature_info: Optional[Dict[str, Any]] = None,
    num_classes: Optional[Sequence[int]] = None,
    *,
    num_steps: int = 6000,
    n_estimators: int = 250,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    One evaluation fold:

    diffusion <- train_diffusion(train)
    X_bal, y_bal <- undersample(diffusion, train)
    clf <- train_classifier(X_bal, y_bal)
    return test AUC and probabilities
    """
    diffusion = train_diffusion(
        X_train, y_train, feature_info, num_classes,
        num_steps=num_steps, seed=seed,
    )
    X_bal, y_bal = undersample(diffusion, X_train, y_train)
    clf = train_classifier(X_bal, y_bal, n_estimators=n_estimators, seed=seed)

    X_te = _to_numpy(X_test)
    y_te = _to_numpy(y_test).reshape(-1)
    proba = clf.predict_proba(X_te)
    pos_idx = list(clf.classes_).index(MINORITY_LABEL) if MINORITY_LABEL in clf.classes_ else 1
    y_score = proba[:, pos_idx]
    auc = float("nan") if len(np.unique(y_te)) < 2 else float(roc_auc_score(y_te, y_score))
    return {
        "auc": auc,
        "y_score": y_score,
        "n_balanced": int(len(y_bal)),
        "classifier": clf,
        "diffusion": diffusion,
    }


def _synthetic_imbalance(n_maj: int = 200, n_min: int = 20, d: int = 8, seed: int = 0):
    rng = np.random.RandomState(seed)
    X_maj = rng.normal(0.0, 1.0, size=(n_maj, d))
    X_min = rng.normal(1.2, 1.0, size=(n_min, d))
    X = np.vstack([X_maj, X_min]).astype(np.float32)
    y = np.concatenate([np.zeros(n_maj, dtype=int), np.ones(n_min, dtype=int)])
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    cut = int(0.7 * len(y))
    return X[:cut], y[:cut], X[cut:], y[cut:]


if __name__ == "__main__":
    print("Public Diff-REU pipeline (selection module withheld).")
    X_tr, y_tr, X_te, y_te = _synthetic_imbalance()
    feature_info = {"num_numerical": X_tr.shape[1]}
    handle = train_diffusion(X_tr, y_tr, feature_info=feature_info, num_steps=100, seed=0)
    print("  diffusion handle:", handle.n_features, "features, majority class", handle.majority_class)
    try:
        out = run_one_fold(X_tr, y_tr, X_te, y_te, feature_info=feature_info, num_steps=100, seed=0)
        print("  fold AUC = %.4f" % out["auc"])
    except RuntimeError as err:
        print("  undersample() is a black box in this release:")
        print("   ", str(err).split("The full implementation")[0].strip())
        print("  Plug in diffreu_core.py to run the full fold.")
