"""
Trains the model shipped with the Streamlit demo app.

Uses Model A (conservative feature set, no STR_*/ING_* strike-context
columns) with the team's own class_weight dict, so the demo works
out-of-the-box with no extra dependencies (no xgboost/lightgbm needed).

Swap in the team's actual tuned model later: save it as demo_model.joblib
with the same 80-column input order as model_data_A_v2_imputed_X_train.csv
and the app will use it without any other changes.
"""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

X_train = pd.read_csv("model_data_A_v2_imputed_X_train.csv")
y_train = pd.read_csv("model_data_A_v2_imputed_y_train.csv").iloc[:, 0]
X_test = pd.read_csv("model_data_A_v2_imputed_X_test.csv")
y_test = pd.read_csv("model_data_A_v2_imputed_y_test.csv").iloc[:, 0]
class_weight = joblib.load("model_data_A_v2_imputed_class_weight.joblib")

model = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["No Damage", "Damage"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"PR-AUC:  {average_precision_score(y_test, y_proba):.4f}")

joblib.dump(model, "demo_model.joblib")
joblib.dump(list(X_train.columns), "demo_model_columns.joblib")
print("\nSaved demo_model.joblib + demo_model_columns.joblib")
