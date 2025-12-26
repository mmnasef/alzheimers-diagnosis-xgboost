import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
import os
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

os.makedirs("results/shap_plots/dependence", exist_ok=True)

df = pd.read_csv("AIBL.csv")

df["Age"] = df["Examyear"] - df["PTDOBYear"]

selected_features = [
    "CDGLOBAL", "Age", "PTGENDER",
    "APGEN1", "APGEN2",
    "RCT392", "RCT6", "RCT20", "HMT3", "HMT40",
    "MH12RENA", "MH2NEURL", "MH16SMOK"
]

diagnosis_matrix = df[["DXNORM", "DXMCI", "DXAD"]].values
y = pd.Series(diagnosis_matrix.argmax(axis=1))

X = df[selected_features]

class_names = ["Normal", "MCI", "Alzheimer's"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

explainer = shap.TreeExplainer(
    model,
    X_train,
    feature_perturbation="interventional"
)

shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, np.ndarray):
    shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

for i, class_name in enumerate(class_names):
    shap.summary_plot(
        shap_values[i],
        X_test,
        show=False
    )
    plt.title(f"SHAP Summary – {class_name}")
    plt.savefig(
        f"results/shap_plots/summary_{class_name.lower().replace(' ', '_')}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

for i, class_name in enumerate(class_names):
    shap.summary_plot(
        shap_values[i],
        X_test,
        plot_type="bar",
        show=False
    )
    plt.title(f"Feature Importance – {class_name}")
    plt.savefig(
        f"results/shap_plots/bar_{class_name.lower().replace(' ', '_')}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

examples = {}
for i in range(3):
    mask = (y_test.values == i) & (y_pred == i)
    if mask.sum() > 0:
        examples[i] = np.where(mask)[0][0]

for class_idx, idx in examples.items():
    base_val = explainer.expected_value[class_idx]
    explanation = shap.Explanation(
        values=shap_values[class_idx][idx],
        base_values=base_val,
        data=X_test.iloc[idx].values,
        feature_names=selected_features
    )
    shap.waterfall_plot(explanation, show=False)
    plt.savefig(
        f"results/shap_plots/waterfall_{class_names[class_idx].lower().replace(' ', '_')}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

ad_importance = np.abs(shap_values[2]).mean(axis=0)
top_idx = np.argsort(ad_importance)[-3:][::-1]
top_features = [selected_features[i] for i in top_idx]

for feature in top_features:
    shap.dependence_plot(
        feature,
        shap_values[2],
        X_test,
        show=False
    )
    plt.savefig(
        f"results/shap_plots/dependence/{feature}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

importance_stats = {}
for i, class_name in enumerate(class_names):
    importance_stats[class_name] = dict(
        zip(
            selected_features,
            np.abs(shap_values[i]).mean(axis=0)
        )
    )

importance_df = pd.DataFrame(importance_stats)
importance_df = importance_df.sort_values("Alzheimer's", ascending=False)
importance_df.to_csv("results/shap_feature_importance.csv")

results = {
    "accuracy": float(accuracy),
    "method": "TreeExplainer (interventional)",
    "features": selected_features,
    "class_distribution": {
        class_names[i]: int((y == i).sum()) for i in range(3)
    },
    "class_metrics": {
        class_name: {
            "precision": float(report[class_name]["precision"]),
            "recall": float(report[class_name]["recall"]),
            "f1": float(report[class_name]["f1-score"]),
            "support": int(report[class_name]["support"])
        } for class_name in class_names
    },
    "top_3_ad_features": {
        "features": top_features,
        "importance": [float(ad_importance[selected_features.index(f)]) for f in top_features]
    }
}

with open("results/shap_analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("SHAP analysis completed successfully")
print(importance_df.head())
