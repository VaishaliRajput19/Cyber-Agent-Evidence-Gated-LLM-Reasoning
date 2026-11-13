"""Hybrid threat detection (Section III-B of the paper).

Binary XGBoost gives p_i, multiclass XGBoost gives q_i over K attack
categories, and Isolation Forest gives the anomaly score s_i (higher = more
anomalous). The union fusion rule (Eq. 4) and the evidence label (Eq. 5) are
implemented here, together with training and loading of the three models.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import List, Sequence

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from .egar import ScoreNormalizer

try:
    from xgboost import XGBClassifier

    def _xgb(num_class: int = 2):
        params = dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                      subsample=0.9, colsample_bytree=0.9, tree_method="hist")
        if num_class > 2:
            return XGBClassifier(objective="multi:softprob", num_class=num_class, **params)
        return XGBClassifier(objective="binary:logistic", **params)
except ImportError:  # fallback so the pipeline can still be tested
    from sklearn.ensemble import HistGradientBoostingClassifier

    def _xgb(num_class: int = 2):
        warnings.warn("xgboost not installed: using sklearn HistGradientBoosting instead")
        return HistGradientBoostingClassifier(max_iter=300)


@dataclass
class HybridDetector:
    binary: object
    multiclass: object
    iforest: IsolationForest
    categories: List[str]
    normalizer: ScoreNormalizer
    tau: float                    # Isolation Forest threshold on s_i

    # --- detector outputs used by EGAR ------------------------------------
    def f_b(self, x: Sequence[float]) -> float:
        return float(self.binary.predict_proba(np.asarray([x]))[0][1])

    def f_m(self, x: Sequence[float]) -> List[float]:
        return list(self.multiclass.predict_proba(np.asarray([x]))[0])

    def f_if(self, x: Sequence[float]) -> float:
        return float(-self.iforest.score_samples(np.asarray([x]))[0])

    @property
    def tau_tilde(self) -> float:
        return self.normalizer(self.tau)

    # --- Eq. (4) and Eq. (5) ---------------------------------------------
    def fused(self, x: Sequence[float]) -> int:
        return int(self.f_b(x) >= 0.5 or self.f_if(x) > self.tau)

    def evidence(self, x: Sequence[float]) -> str:
        if self.f_b(x) >= 0.5:
            q = self.f_m(x)
            return self.categories[int(np.argmax(q))]
        if self.f_if(x) > self.tau:
            return "unknown-anomalous"
        return "benign"

    # --- persistence --------------------------------------------------------
    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.binary, os.path.join(out_dir, "xgb_binary.joblib"))
        joblib.dump(self.multiclass, os.path.join(out_dir, "xgb_multiclass.joblib"))
        joblib.dump(self.iforest, os.path.join(out_dir, "iforest.joblib"))
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump({"categories": self.categories, "tau": self.tau,
                       "s_min": self.normalizer.s_min, "s_max": self.normalizer.s_max}, f, indent=2)

    @classmethod
    def load(cls, model_dir: str) -> "HybridDetector":
        with open(os.path.join(model_dir, "meta.json")) as f:
            meta = json.load(f)
        norm = ScoreNormalizer()
        norm.s_min, norm.s_max = meta["s_min"], meta["s_max"]
        return cls(
            binary=joblib.load(os.path.join(model_dir, "xgb_binary.joblib")),
            multiclass=joblib.load(os.path.join(model_dir, "xgb_multiclass.joblib")),
            iforest=joblib.load(os.path.join(model_dir, "iforest.joblib")),
            categories=meta["categories"], normalizer=norm, tau=meta["tau"])


def train_hybrid(X: np.ndarray, y: Sequence[str], contamination: float = 0.05,
                 seed: int = 42) -> HybridDetector:
    """Train the three detectors. `y` holds labels: "benign" or an attack category."""
    y = np.asarray(y)
    malicious = (y != "benign").astype(int)

    binary = _xgb(2)
    binary.fit(X, malicious)

    categories = sorted(set(y[malicious == 1]))
    idx = {c: i for i, c in enumerate(categories)}
    Xm = X[malicious == 1]
    ym = np.asarray([idx[c] for c in y[malicious == 1]])
    multiclass = _xgb(len(categories))
    multiclass.fit(Xm, ym)

    # Isolation Forest learns normal behavior from benign traffic only
    iforest = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    iforest.fit(X[malicious == 0])
    tau = float(-iforest.offset_)            # anomalous when -score_samples > tau
    train_scores = -iforest.score_samples(X)
    norm = ScoreNormalizer().fit(train_scores)
    return HybridDetector(binary, multiclass, iforest, categories, norm, tau)
