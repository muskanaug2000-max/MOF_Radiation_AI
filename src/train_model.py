from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


def train(df):

    df = df.fillna("Unknown")

    encoders = {}

    categorical = [
        "MOF",
        "Metal",
        "Linker",
        "Radiation Type",
        "Source",
        "Atmosphere",
        "Stability"
    ]

    for col in categorical:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    X = df[[
        "MOF",
        "Metal",
        "Radiation Type"
    ]]

    y = df["Stability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, pred))

    return model