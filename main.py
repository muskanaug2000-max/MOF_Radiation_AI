from src.read_data import load_data
from src.train_model import train

df = load_data()

print(df.head())

print("Dataset Shape:", df.shape)

model = train(df)