import pandas as pd

df = pd.read_csv("data/factoid/proposition.csv", chunksize=100000000)

for i, chunk in enumerate(df):
    print(f"Saving chunk {i}...")
    chunk.to_parquet(f"proposition{i}.parquet")
