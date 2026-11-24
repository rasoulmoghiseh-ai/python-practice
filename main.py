# Welcome to Python Compiler! Press Ctrl+Enter to run
print("Hello, World!")

# Pandas example
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries loaded successfully!")
import pandas as pd
import numpy as np

# Telco-Customer-Churn.csv

import pandas as pd
import numpy as np

# خواندن دیتاست
df = pd.read_csv("Telco-Customer-Churn.csv")

# نمایش 5 مشتری اول
print("First 5 Customers:")
print(df.head())

# ابعاد دیتاست
print("\nDataset Shape:")
print(df.shape)

# اطلاعات ستون ها
print("\nDataset Information:")
df.info()

# بررسی داده های گمشده
print("\nMissing Values:")
print(df.isnull().sum())

# آمار توصیفی
print("\nStatistical Description:")
print(df.describe())
# تبدیل TotalCharges به عدد
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# بررسی مقادیر گمشده جدید
print("Missing TotalCharges:")
print(df["TotalCharges"].isnull().sum())

# بررسی ردیف‌های مشکل‌دار
print(df[df["TotalCharges"].isnull()][
    ["tenure", "MonthlyCharges", "TotalCharges"]
])
# تعداد مشتریان در هر گروه
print(df["Churn"].value_counts())

# درصد مشتریان در هر گروه
print(df["Churn"].value_counts(normalize=True) * 100)

# رسم نمودار
sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()
churn_contract = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print(churn_contract)

churn_contract.plot(
    kind="bar",
    stacked=True
)

plt.title("Churn Rate by Contract Type")
plt.ylabel("Percentage (%)")
plt.xlabel("Contract Type")
plt.show()
sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Tenure Distribution by Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# نمودار مدت اشتراک
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Tenure vs Churn")
plt.show()

# نمودار هزینه ماهانه
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Churn")
plt.show()
from scipy.stats import mannwhitneyu

# Separate the two groups
churn_yes = df[df["Churn"] == "Yes"]
churn_no = df[df["Churn"] == "No"]

# Test 1: Monthly Charges
stat1, p1 = mannwhitneyu(
    churn_yes["MonthlyCharges"],
    churn_no["MonthlyCharges"]
)

# Test 2: Tenure
stat2, p2 = mannwhitneyu(
    churn_yes["tenure"],
    churn_no["tenure"]
)

print("Monthly Charges:")
print("P-value:", p1)

print("\nTenure:")
print("P-value:", p2)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# 1. Load and clean data
df = pd.read_csv("Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df = df.drop(columns=["customerID"])

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 2. Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

# 3. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
}

# 6. Train and evaluate
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    print("\n==============================")
    print(name)
    print("==============================")

    print("Accuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))
    print("Precision:", round(
        precision_score(y_test, predictions), 4
    ))
    print("Recall:", round(
        recall_score(y_test, predictions), 4
    ))
    print("F1-score:", round(
        f1_score(y_test, predictions), 4
    ))
    print("ROC-AUC:", round(
        roc_auc_score(y_test, probabilities), 4
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions,
        target_names=["No Churn", "Churn"]
    ))
  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# =====================================
# 1. Load and prepare data
# =====================================

df = pd.read_csv("Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df = df.drop(columns=["customerID"])

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

# =====================================
# 2. Split into training, validation, test
# =====================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2, random_state=42, stratify=y_train_full
)

# =====================================
# 3. Train model for threshold selection
# =====================================

validation_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ))
])

validation_model.fit(X_train, y_train)

val_probabilities = validation_model.predict_proba(X_val)[:, 1]

# Find threshold with best validation F1
thresholds = np.arange(0.20, 0.81, 0.01)

best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:
    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    score = f1_score(y_val, val_predictions)

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

print("Best Threshold:", round(best_threshold, 2))
print("Best Validation F1:", round(best_f1, 4))

# =====================================
# 4. Train final model on all training data
# =====================================

final_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ))
])

final_model.fit(X_train_full, y_train_full)

test_probabilities = final_model.predict_proba(X_test)[:, 1]

# Default threshold
pred_default = (test_probabilities >= 0.5).astype(int)

# Tuned threshold
pred_tuned = (
    test_probabilities >= best_threshold
).astype(int)

# =====================================
# 5. Compare thresholds
# =====================================

for name, predictions in [
    ("Default Threshold (0.5)", pred_default),
    ("Tuned Threshold", pred_tuned)
]:
    print("\n==============================")
    print(name)
    print("==============================")

    print("Accuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))
    print("Precision:", round(
        precision_score(y_test, predictions), 4
    ))
    print("Recall:", round(
        recall_score(y_test, predictions), 4
    ))
    print("F1-score:", round(
        f1_score(y_test, predictions), 4
    ))
    print("ROC-AUC:", round(
        roc_auc_score(y_test, test_probabilities), 4
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

# =====================================
# 6. Feature importance using coefficients
# =====================================

fitted_preprocessor = final_model.named_steps["preprocessing"]
logistic_model = final_model.named_steps["model"]

feature_names = fitted_preprocessor.get_feature_names_out()
coefficients = logistic_model.coef_[0]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

importance_df["Absolute_Importance"] = (
    importance_df["Coefficient"].abs()
)

importance_df = importance_df.sort_values(
    by="Absolute_Importance", ascending=False
)

print("\nTop 15 Important Features:")
print(importance_df.head(15).to_string(index=False))

# =====================================
# 7. Plot feature coefficients
# =====================================

top_features = importance_df.head(15).sort_values(
    by="Coefficient"
)

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["Feature"],
    top_features["Coefficient"]
)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Top 15 Logistic Regression Coefficients")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Matplotlib example (uncomment to run)
# import matplotlib.pyplot as plt
# plt.plot([1, 2, 3], [1, 4, 9])
# plt.title('Simple Plot')
# plt.show()
