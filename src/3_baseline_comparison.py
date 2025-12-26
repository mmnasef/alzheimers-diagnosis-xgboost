import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

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

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", random_state=42),
    "XGBoost": xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        learning_rate=0.03,
        max_depth=4,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "accuracy": float(acc),
        "precision": float(report["weighted avg"]["precision"]),
        "recall": float(report["weighted avg"]["recall"]),
        "f1": float(report["weighted avg"]["f1-score"]),
        "confusion_matrix": cm.tolist()
    }

os.makedirs("results", exist_ok=True)

with open("results/comparison_results.json", "w") as f:
    json.dump(results, f, indent=4)

model_names = list(results.keys())
accuracies = [results[m]["accuracy"] for m in model_names]
precisions = [results[m]["precision"] for m in model_names]
recalls = [results[m]["recall"] for m in model_names]
f1s = [results[m]["f1"] for m in model_names]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
bars = axes[0].bar(model_names, accuracies, color=colors)
axes[0].set_ylim([0.7, 1.0])

for bar in bars:
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}", ha="center")

x = np.arange(len(model_names))
w = 0.2

axes[1].bar(x - 1.5*w, accuracies, w, label="Accuracy")
axes[1].bar(x - 0.5*w, precisions, w, label="Precision")
axes[1].bar(x + 0.5*w, recalls, w, label="Recall")
axes[1].bar(x + 1.5*w, f1s, w, label="F1")

axes[1].set_xticks(x)
axes[1].set_xticklabels(model_names, rotation=15)
axes[1].legend()

plt.tight_layout()
plt.savefig("results/baseline_comparison.png", dpi=300)
plt.close()

sorted_models = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)

best_model = sorted_models[0][0]
best_acc = sorted_models[0][1]["accuracy"]
second_acc = sorted_models[1][1]["accuracy"]

print(f"Best Model: {best_model}")
print(f"Accuracy: {best_acc*100:.2f}%")
print(f"Improvement: {(best_acc-second_acc)*100:.2f}%")
