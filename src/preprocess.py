import pandas as pd

def load_data(path, sample_size=None):
    # Load dataset from CSV file
    df = pd.read_csv(path)

    # Optional sampling for development/testing
    if sample_size is not None:
        df = df.sample(min(sample_size, len(df)), random_state=42)
    return df

def clean_data(df):
    # Remove rows with missing values
    df = df.dropna()
    threshold = df["PINCP"].median()
    # Convert 'PINCP' to binary based on median threshold
    df["PINCP"] = (df["PINCP"] > threshold).astype(int)

    return df