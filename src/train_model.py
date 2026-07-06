import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/mof_radiation.csv")

print("Dataset:")
print(df)

# Convert text to numbers
df["Metal"] = df["metal"].map({"Zr": 0, "Zn": 1})
df["Radiation"] = df["radiation_type"].map({"Gamma": 0, "Electron": 1})

# Features (X)
X = df[["Metal", "Radiation", "dose_kGy"]]

# Target (y)
y = df["stability"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

print("\nModel trained successfully!")

# Predict a new MOF
new_sample = pd.DataFrame({
    "Metal": [0],
    "Radiation": [0],
    "dose_kGy": [800]
})

prediction = model.predict(new_sample)
print(f"\nPredicted Stability = {prediction[0]:.3f}")