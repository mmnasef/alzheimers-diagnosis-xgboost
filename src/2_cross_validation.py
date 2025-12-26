import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import os
from datetime import datetime

df = pd.read_csv("AIBL.csv")

df["DX"] = np.select(
    [df["DXNORM"] == 1, df["DXMCI"] == 1, df["DXAD"] == 1],
    [0, 1, 2]
)

df = df.drop(columns=["DXNORM", "DXMCI", "DXAD", "DXCURREN"])
df["Age"] = df["Examyear"] - df["PTDOBYear"]
df = df.drop(columns=["Examyear", "APTyear", "PTDOBYear"])

features = [
    "CDGLOBAL", "Age", "PTGENDER",
    "APGEN1", "APGEN2",
    "RCT392", "RCT6", "RCT20", "HMT3", "HMT40",
    "MH12RENA", "MH2NEURL", "MH16SMOK"
]

X = df[features]
y = df["DX"]

params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "seed": 42
}

weights = {0: 1.0, 1: 4.47, 2: 6.24}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = []
all_preds = []
all_true = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    sample_weights = y_train.map(weights)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    y_pred = model.predict(dval).argmax(axis=1)

    acc = accuracy_score(y_val, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="weighted", zero_division=0
    )

    prec_pc, rec_pc, f1_pc, _ = precision_recall_fscore_support(
        y_val, y_pred, average=None, zero_division=0
    )

    fold_metrics.append({
        "fold": fold,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "per_class": {
            "Normal": {"precision": float(prec_pc[0]), "recall": float(rec_pc[0]), "f1": float(f1_pc[0])},
            "MCI": {"precision": float(prec_pc[1]), "recall": float(rec_pc[1]), "f1": float(f1_pc[1])},
            "AD": {"precision": float(prec_pc[2]), "recall": float(rec_pc[2]), "f1": float(f1_pc[2])}
        }
    })

    all_preds.extend(y_pred.tolist())
    all_true.extend(y_val.tolist())

accuracies = [m["accuracy"] for m in fold_metrics]
precisions = [m["precision"] for m in fold_metrics]
recalls = [m["recall"] for m in fold_metrics]
f1s = [m["f1"] for m in fold_metrics]

stats = {
    "overall": {
        "accuracy": {
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies)),
            "min": float(np.min(accuracies)),
            "max": float(np.max(accuracies))
        },
        "precision": {
            "mean": float(np.mean(precisions)),
            "std": float(np.std(precisions))
        },
        "recall": {
            "mean": float(np.mean(recalls)),
            "std": float(np.std(recalls))
        },
        "f1": {
            "mean": float(np.mean(f1s)),
            "std": float(np.std(f1s))
        }
    }
}

os.makedirs("results", exist_ok=True)

output = {
    "metadata": {
        "method": "5-Fold Stratified CV",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "samples": len(X)
    },
    "statistics": stats,
    "fold_metrics": fold_metrics
}

with open("results/cv_results.json", "w") as f:
    json.dump(output, f, indent=4)

print(f"CV Accuracy: {stats['overall']['accuracy']['mean']*100:.2f}% ± {stats['overall']['accuracy']['std']*100:.2f}%")
