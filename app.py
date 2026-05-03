from src.preprocess import load_data, clean_data
from src.train import train_model

def main():

    print("Loading data...")
    df = load_data("data/dataset.csv", sample_size=150000)

    print("Cleaning data...")
    df = clean_data(df)

    print("Shape:", df.shape)

    print("Training model...")
    model, acc, cm, report = train_model(df, target="PINCP")

    print("Training pipeline completed successfully ✔")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:", cm)
    print("Classification report:", report)

    print("Model trained:", model)

if __name__ == "__main__":
    main()