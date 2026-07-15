import pandas as pd

def load_data():
    df = pd.read_csv("data/mof_radiation.csv")
    print(df.head())
    print("\nDataset Shape:", df.shape)
    return df