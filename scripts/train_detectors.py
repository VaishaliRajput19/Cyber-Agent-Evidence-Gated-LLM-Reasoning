"""Train the hybrid detectors and report detection results (Section V-A).

  python scripts/train_detectors.py --data "data/flows/*.csv" --holdout ddos --out models/

With --holdout ddos, DDoS flows are removed from supervised training and used
only to measure unseen-attack detection (XGBoost alone, Isolation Forest
alone, and the fused detector of Eq. 4).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cyber_agent.detection import train_hybrid  # noqa: E402
from cyber_agent.features import FEATURE_NAMES  # noqa: E402


def metrics(y_true, y_pred, average):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=average, zero_division=0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="glob of flow CSV files")
    ap.add_argument("--holdout", default=None, help="attack category withheld from training")
    ap.add_argument("--contamination", type=float, default=0.05)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(p) for p in glob.glob(args.data)], ignore_index=True)
    held = df[df.label == args.holdout] if args.holdout else df.iloc[0:0]
    df = df[df.label != args.holdout] if args.holdout else df
    X, y = df[FEATURE_NAMES].to_numpy(float), df.label.to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)

    det = train_hybrid(Xtr, ytr, contamination=args.contamination)
    det.save(args.out)

    bin_true = (yte != "benign").astype(int)
    p = det.binary.predict_proba(Xte)[:, 1]
    s = -det.iforest.score_samples(Xte)
    res = {"binary_xgboost": metrics(bin_true, (p >= 0.5).astype(int), "binary"),
           "isolation_forest": metrics(bin_true, (s > det.tau).astype(int), "macro")}
    mal = bin_true == 1
    mc_pred = np.asarray(det.categories)[det.multiclass.predict_proba(Xte[mal]).argmax(1)]
    res["multiclass_xgboost"] = metrics(yte[mal], mc_pred, "macro")

    if len(held):
        Xh = held[FEATURE_NAMES].to_numpy(float)
        ph = det.binary.predict_proba(Xh)[:, 1] >= 0.5
        sh = -det.iforest.score_samples(Xh) > det.tau
        res["unseen_" + args.holdout] = {"xgboost": float(ph.mean()),
                                         "isolation_forest": float(sh.mean()),
                                         "fused": float((ph | sh).mean())}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "detection_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
