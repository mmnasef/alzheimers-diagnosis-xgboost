import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import xgboost as xgb
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

weights = {0: 1.0, 1: 4.47, 2: 6.24}
sample_weights = y_train.map(weights)

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

dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
dtest = xgb.DMatrix(X_test, label=y_test)

evals_result = {}
model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, "train"), (dtest, "test")],
    evals_result=evals_result,
    early_stopping_rounds=30,
    verbose_eval=False
)

y_pred = model.predict(dtest).argmax(axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted"
)

cm = confusion_matrix(y_test, y_pred)

results = {
    "metadata": {
        "model": "XGBoost Baseline",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    },
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(
        y_test, y_pred,
        target_names=["Normal", "MCI", "AD"],
        output_dict=True
    )
}

os.makedirs("results", exist_ok=True)
with open("results/baseline_results.json", "w") as f:
    json.dump(results, f, indent=4)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Normal", "MCI", "AD"],
    yticklabels=["Normal", "MCI", "AD"],
    ax=axes[0, 0]
)

train_loss = evals_result["train"]["mlogloss"]
test_loss = evals_result["test"]["mlogloss"]

axes[0, 1].plot(train_loss, label="Train")
axes[0, 1].plot(test_loss, label="Test")
axes[0, 1].axvline(model.best_iteration, linestyle="--", color="red")
axes[0, 1].legend()

importance = model.get_score(importance_type="weight")
imp_df = pd.DataFrame({
    "Feature": list(importance.keys()),
    "Importance": list(importance.values())
}).sort_values("Importance")

axes[1, 0].barh(imp_df["Feature"], imp_df["Importance"])

precision_c, recall_c, f1_c, _ = precision_recall_fscore_support(y_test, y_pred)
x = np.arange(3)
w = 0.25

axes[1, 1].bar(x - w, precision_c, w, label="Precision")
axes[1, 1].bar(x, recall_c, w, label="Recall")
axes[1, 1].bar(x + w, f1_c, w, label="F1")
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(["Normal", "MCI", "AD"])
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("results/baseline_evaluation.png", dpi=300)
plt.close()

os.makedirs("models", exist_ok=True)
model.save_model("models/baseline_model.json")

print(f"Accuracy: {accuracy*100:.2f}%")
