"""Classification metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.

Usage:
    python classification_metrics.py      # runs the demo
    from classification_metrics import evaluate_classification
"""
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)


def evaluate_classification(y_true, y_pred, y_proba=None, average="binary"):
    """Return a dict of classification scores.

    y_true, y_pred : true and predicted labels
    y_proba        : probability of the positive class (binary only), used for ROC-AUC
    average        : "binary", "micro", "macro" or "weighted" (multi-class averaging)
    """
    scores = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    if y_proba is not None:
        scores["roc_auc"] = roc_auc_score(y_true, y_proba)
    return scores


def f1_averages(y_true, y_pred):
    """Micro, macro and weighted F1 for multi-class problems."""
    return {a: f1_score(y_true, y_pred, average=a, zero_division=0) for a in ("micro", "macro", "weighted")}


def accuracy_trap(n_negative=95, n_positive=5):
    """Show that a model always predicting the majority class gets high accuracy but F1 = 0."""
    y_true = [0] * n_negative + [1] * n_positive
    y_pred = [0] * (n_negative + n_positive)
    return {"accuracy": accuracy_score(y_true, y_pred), "f1": f1_score(y_true, y_pred, zero_division=0)}


if __name__ == "__main__":
    # Binary example: 10 reviews, 1 = positive, 0 = negative
    y_true = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
    y_proba = [0.95, 0.9, 0.8, 0.7, 0.45, 0.3, 0.2, 0.1, 0.35, 0.6]
    print("Binary scores    :", {k: round(v, 3) for k, v in evaluate_classification(y_true, y_pred, y_proba).items()})
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_true, y_pred))

    print("\nAccuracy trap (95 negatives / 5 positives, always 'negative'):", accuracy_trap())

    y_true_m = ["sports"] * 6 + ["politics"] * 2 + ["tech"] * 2
    y_pred_m = ["sports"] * 7 + ["politics", "tech", "sports"]
    print("\nMulti-class F1   :", {k: round(v, 3) for k, v in f1_averages(y_true_m, y_pred_m).items()})
    print(classification_report(y_true_m, y_pred_m, zero_division=0))
