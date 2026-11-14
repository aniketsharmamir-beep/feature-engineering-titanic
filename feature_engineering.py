"""
Feature engineering and selection on the cleaned Titanic dataset.

Steps:
1. Create new features from existing columns
2. Encode categorical variables (one-hot encoding)
3. Scale numerical features (standardization)
4. Select the most relevant features (ANOVA F-test)
5. Compare a model trained on all features vs. selected features
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/cleaned_dataset.csv"
OUTPUT_PATH = "data/engineered_features.csv"
K_FEATURES = 10  # how many features to keep


def create_features(df):
    """Create new features from the existing columns."""
    df = df.copy()

    # Family size = siblings/spouses + parents/children + the passenger
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # 1 if the passenger travelled alone, otherwise 0
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Fare per person (the ticket price is often shared by the family)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    # Extract the title (Mr, Mrs, Miss...) from the name
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    common_titles = ["Mr", "Mrs", "Miss", "Master"]
    df["Title"] = df["Title"].where(df["Title"].isin(common_titles), "Rare")

    # Group ages into categories
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 60, 100],
        labels=["Child", "Teen", "Adult", "Senior"],
    )

    # Drop columns that are only identifiers
    return df.drop(columns=["PassengerId", "Name", "Ticket"])


def encode_categoricals(df):
    """Turn text columns into 0/1 columns (one-hot encoding)."""
    categorical_cols = ["Sex", "Embarked", "Title", "AgeGroup"]
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)


def main():
    df = pd.read_csv(DATA_PATH)
    print("Original shape:", df.shape)

    df = create_features(df)
    df = encode_categoricals(df)
    print("Shape after feature engineering:", df.shape)

    # Save the full engineered dataset
    df.to_csv(OUTPUT_PATH, index=False)

    X = df.drop(columns="Survived")
    y = df["Survived"]

    # Split first, so scaling and selection only learn from the training data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the numerical columns (mean 0, standard deviation 1)
    numeric_cols = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "FarePerPerson"]
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Feature selection: keep the K features most related to Survived
    selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
    selector.fit(X_train, y_train)

    scores = pd.Series(selector.scores_, index=X_train.columns)
    print("\nFeature scores (higher = more related to survival):")
    print(scores.sort_values(ascending=False).round(1))

    selected = list(X_train.columns[selector.get_support()])
    print("\nSelected features:", selected)

    # Compare a model using all features vs. only the selected ones
    model_all = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    acc_all = accuracy_score(y_test, model_all.predict(X_test))

    model_sel = LogisticRegression(max_iter=1000).fit(X_train[selected], y_train)
    acc_sel = accuracy_score(y_test, model_sel.predict(X_test[selected]))

    print(f"\nAccuracy with all {X.shape[1]} features: {acc_all:.3f}")
    print(f"Accuracy with {K_FEATURES} selected features: {acc_sel:.3f}")

    # Try different numbers of features to see how accuracy changes
    print("\nAccuracy for different numbers of selected features:")
    for k in range(4, X.shape[1] + 1, 2):
        cols = list(X_train.columns[SelectKBest(f_classif, k=k).fit(X_train, y_train).get_support()])
        model = LogisticRegression(max_iter=1000).fit(X_train[cols], y_train)
        acc = accuracy_score(y_test, model.predict(X_test[cols]))
        print(f"  k={k:2d} -> accuracy {acc:.3f}")


if __name__ == "__main__":
    main()
