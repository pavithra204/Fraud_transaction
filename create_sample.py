"""
Run this script ONCE after downloading creditcard.csv from Kaggle.
It creates a small sample (5000 rows) that can be pushed to GitHub.

Usage:
    python create_sample.py

Input  : creditcard.csv  (download from Kaggle — 144 MB)
Output : sample_creditcard.csv  (push this to GitHub — ~5 MB)
"""

import pandas as pd
import os

INPUT  = "creditcard.csv"
OUTPUT = "sample_creditcard.csv"

if not os.path.exists(INPUT):
    print(f"ERROR: '{INPUT}' not found.")
    print("Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud")
    exit(1)

df = pd.read_csv(INPUT)
print(f"Full dataset : {len(df):,} rows | Fraud: {df['Class'].sum()} cases")

# Keep all fraud + sample of normal so fraud isn't lost
fraud  = df[df['Class'] == 1]                        # all 492 fraud rows
normal = df[df['Class'] == 0].sample(n=4508, random_state=42)

sample = pd.concat([fraud, normal]).sample(frac=1, random_state=42).reset_index(drop=True)
sample.to_csv(OUTPUT, index=False)

print(f"Sample saved : {len(sample):,} rows | Fraud: {sample['Class'].sum()} cases")
print(f"File         : {OUTPUT}")
print(f"Size         : {os.path.getsize(OUTPUT) / 1e6:.2f} MB")
print("\nNow push sample_creditcard.csv to your GitHub repo!")
