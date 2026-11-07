import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")

# ---------- 13. Load data ----------
print("=== Step 13: Loading train.csv ===")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
print(f"Train shape: {train.shape}")
print(train.head())

# ---------- 14. Explore & clean ----------
print("\n=== Step 14: Exploring & cleaning data ===")
print("Missing values per column:\n", train.isnull().sum())

age_median = train["Age"].median()
fare_median = train["Fare"].median()
embarked_mode = train["Embarked"].mode()[0]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age"] = df["Age"].fillna(age_median)
    df["Fare"] = df["Fare"].fillna(fare_median)
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df = pd.get_dummies(df, columns=["Embarked"], prefix="Emb")
    for col in ["Emb_C", "Emb_Q", "Emb_S"]:
        if col not in df:
            df[col] = 0
    return df


print(f"Filled Age NA with median = {age_median}")
print(f"Filled Fare NA with median = {fare_median}")
print(f"Filled Embarked NA with mode = {embarked_mode}")
print("Encoded Sex (male=0, female=1); one-hot encoded Embarked")
print("Added feature FamilySize = SibSp + Parch + 1")
print("Dropped Cabin, Name, Ticket (too many missing / not numeric)")

train_p = prepare(train)
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
            "FamilySize", "Emb_C", "Emb_Q", "Emb_S"]
X = train_p[FEATURES].astype(float)
y = train_p["Survived"]

# ---------- 15. Train model ----------
print("\n=== Step 15: Training logistic regression ===")
print(f"Features used: {FEATURES}")
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000)
model.fit(X_tr, y_tr)
print("Model trained.")

# ---------- 16. Training accuracy ----------
train_acc = accuracy_score(y_tr, model.predict(X_tr))
print(f"\n=== Step 16: Training accuracy: {train_acc:.4f} ===")

# ---------- 17. Predict on test.csv ----------
print("\n=== Step 17: Loading test.csv and predicting ===")
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print(f"Test shape: {test.shape}")
test_p = prepare(test)
test_pred = model.predict(test_p[FEATURES].astype(float))
print(f"Predicted survivors in test set: {test_pred.sum()} / {len(test_pred)}")
print(pd.DataFrame({"PassengerId": test["PassengerId"],
                    "Survived": test_pred}).head(10))

# ---------- 18. Test accuracy ----------
print("\n=== Step 18: Test accuracy ===")
if "Survived" in test.columns:
    print(f"Test accuracy: {accuracy_score(test['Survived'], test_pred):.4f}")
else:
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print("test.csv has no 'Survived' labels, so reporting accuracy on a "
          "20% hold-out split of train.csv instead.")
    print(f"Hold-out accuracy: {val_acc:.4f}")

print("\nDone!")
