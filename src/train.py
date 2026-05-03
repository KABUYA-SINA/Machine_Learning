from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import pickle

from src.metrics import evaluate_model
from src.visualization import plot_confusion_matrix


def train_model(df, target, save_path="models/model.pkl"):

    # Split features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Define model
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Evaluate model
    acc, cm, report = evaluate_model(y_test, preds)

    # Visualize results
    plot_confusion_matrix(cm)

    feature_names = X.columns.tolist()

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump({"model": model,
        "features": feature_names
    }, f)

    # Return everything useful
    return model, acc, cm, report