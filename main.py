import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample


print("Loading data...")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_stata("BDIR81FL.DTA")
df = df.copy()

print("Raw shape:", df.shape)

# -----------------------------
# TARGET CLEANING (EDUCATION)
# -----------------------------
df["v106"] = df["v106"].astype(str).str.lower().str.strip()

edu_map = {
    "no education": 0,
    "less than primary": 0,
    "primary": 1,
    "secondary": 2,
    "higher": 3
}

df["v106_num"] = df["v106"].map(edu_map)
df = df.dropna(subset=["v106_num"])

df["y"] = (df["v106_num"] >= 2).astype(int)

# sanity check
if df["y"].nunique() < 2:
    raise ValueError("Target collapsed to one class. Check v106 mapping.")

# -----------------------------
# REGION (FAIRNESS GROUP)
# -----------------------------
df["region"] = df["v024"].astype(str)

# -----------------------------
# FEATURES (UPDATED + STRONGER)
# -----------------------------
features = ["v012", "v025", "v190"]  # age, urban/rural, wealth index

# safe numeric conversion
df["v190"] = pd.to_numeric(df["v190"], errors="coerce")

X = df[features]
y = df["y"]
groups = df["region"]

print("Class distribution:")
print(y.value_counts())

# -----------------------------
# PREPROCESSING PIPELINE
# -----------------------------
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
    ]
)

# -----------------------------
# GROUP-AWARE SPLIT (NO LEAKAGE)
# -----------------------------
print("Splitting data (group-aware)...")

gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]
g_test = groups.iloc[test_idx]

# -----------------------------
# METRICS
# -----------------------------
def fairness_gap(y_true, y_pred, groups):
    temp = pd.DataFrame({
        "y": y_true,
        "pred": y_pred,
        "g": groups
    })
    err = temp.groupby("g").apply(lambda x: (x["y"] != x["pred"]).mean())
    return err.max() - err.min()


def bootstrap_ci(model, X_test, y_test, groups, n=200):
    gaps = []

    for _ in range(n):
        idx = resample(range(len(y_test)), replace=True)

        X_b = X_test.iloc[idx]
        y_b = y_test.iloc[idx]
        g_b = groups.iloc[idx]

        pred = model.predict(X_b)
        gaps.append(fairness_gap(y_b, pred, g_b))

    return np.percentile(gaps, 2.5), np.percentile(gaps, 97.5)

# -----------------------------
# MODELS
# -----------------------------
logreg = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000))
])

rf = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ))
])

# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate(model, name):
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    gap = fairness_gap(y_test, preds, g_test)
    ci = bootstrap_ci(model, X_test, y_test, g_test)

    print("\nAccuracy:", acc)

    region_error = (
        pd.DataFrame({
            "g": g_test,
            "err": (y_test != preds).astype(int)
        })
        .groupby("g")
        .mean()["err"]
        .sort_values()
    )

    print("\nRegion-wise error:")
    print(region_error)

    print("\nFairness gap:", gap)
    print("Bootstrap 95% CI:", ci)


evaluate(logreg, "Logistic Regression")
evaluate(rf, "Random Forest")

print("\nDONE")
