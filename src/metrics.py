from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(y_true, y_pred, verbose=True):
    # Compute accuracy score
    acc = accuracy_score(y_true, y_pred)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Generate classification report
    report = classification_report(y_true, y_pred)

    # display metrics for debugging or analysis
    if verbose:
        print(f"Accuracy: {acc:.2f}")
        print("\nConfusion Matrix:\n", cm)
        print("\nClassification Report:\n", report)

    return acc, cm, report