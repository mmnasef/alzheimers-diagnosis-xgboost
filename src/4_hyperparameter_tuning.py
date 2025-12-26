import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

os.makedirs("results", exist_ok=True)

df = pd.read_csv("AIBL.csv")

df["Age"] = df["Examyear"] - df["PTDOBYear"]

features = [
    "CDGLOBAL", "Age", "PTGENDER",
    "APGEN1", "APGEN2",
    "RCT392", "RCT6", "RCT20", "HMT3", "HMT40",
    "MH12RENA", "MH2NEURL", "MH16SMOK"
]

y = pd.Series(df[["DXNORM", "DXMCI", "DXAD"]].values.argmax(axis=1))
X = df[features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

baseline_params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "mlogloss"
}

baseline_model = xgb.XGBClassifier(**baseline_params)
baseline_model.fit(X_train, y_train)

baseline_pred = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_pred)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_cv_scores = []

for tr, val in cv.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[tr], X_train.iloc[val]
    y_tr, y_val = y_train.iloc[tr], y_train.iloc[val]
    baseline_model.fit(X_tr, y_tr)
    baseline_cv_scores.append(baseline_model.score(X_val, y_val))

baseline_cv_mean = float(np.mean(baseline_cv_scores))
baseline_cv_std = float(np.std(baseline_cv_scores))

param_grid = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2]
}

total_combinations = np.prod([len(v) for v in param_grid.values()])

xgb_model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    random_state=42,
    eval_metric="mlogloss",
    n_jobs=-1
)

grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

start = time.time()
grid.fit(X_train, y_train)
duration = time.time() - start

best_model = grid.best_estimator_
best_params = grid.best_params_
best_cv_score = float(grid.best_score_)

optimized_pred = best_model.predict(X_test)
optimized_accuracy = accuracy_score(y_test, optimized_pred)

class_names = ["Normal", "MCI", "Alzheimer's"]
report = classification_report(
    y_test,
    optimized_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

cv_results = pd.DataFrame(grid.cv_results_)
top_10 = cv_results.sort_values("rank_test_score").head(10)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(10), top_10["mean_test_score"] * 100)
ax.set_yticks(range(10))
ax.set_yticklabels([f"Rank {i+1}" for i in range(10)])
ax.axvline(baseline_cv_mean * 100, color="r", linestyle="--")
ax.set_xlabel("CV Accuracy (%)")
plt.tight_layout()
plt.savefig("results/hyperparameter_tuning_results.png", dpi=300)
plt.close()

param_importance = {}
for p in ["learning_rate", "max_depth", "subsample", "colsample_bytree"]:
    vals = []
    for v in param_grid[p]:
        m = cv_results["param_" + p] == v
        if m.sum() > 0:
            vals.append(cv_results[m]["mean_test_score"].mean())
    param_importance[p] = float(np.std(vals))

param_importance = dict(sorted(param_importance.items(), key=lambda x: x[1], reverse=True))

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(list(param_importance.keys()), list(param_importance.values()))
plt.tight_layout()
plt.savefig("results/parameter_importance.png", dpi=300)
plt.close()

results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_combinations": int(total_combinations),
    "duration_minutes": round(duration / 60, 2),
    "baseline": {
        "cv_accuracy": baseline_cv_mean,
        "cv_std": baseline_cv_std,
        "test_accuracy": float(baseline_accuracy)
    },
    "optimized": {
        "parameters": best_params,
        "cv_accuracy": best_cv_score,
        "test_accuracy": float(optimized_accuracy)
    },
    "improvement": {
        "cv_accuracy": float(best_cv_score - baseline_cv_mean),
        "test_accuracy": float(optimized_accuracy - baseline_accuracy)
    },
    "class_metrics": {
        c: {
            "precision": float(report[c]["precision"]),
            "recall": float(report[c]["recall"]),
            "f1": float(report[c]["f1-score"]),
            "support": int(report[c]["support"])
        }
        for c in class_names
    },
    "parameter_importance": param_importance
}

with open("results/tuning_results.json", "w") as f:
    json.dump(results, f, indent=2)

best_model.save_model("results/best_tuned_model.json")

print("Hyperparameter tuning completed successfully")
print(f"Baseline CV: {baseline_cv_mean*100:.2f}%")
print(f"Optimized CV: {best_cv_score*100:.2f}%")
print(f"Improvement: {(best_cv_score - baseline_cv_mean)*100:+.2f}%")
