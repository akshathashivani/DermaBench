import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def evaluate_model(
    model,
    test_loader,
    device,
    class_names,
    output_dir="reports/figures"
):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)

    # Calculate evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
    }

    try:
        metrics["roc_auc"] = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
            labels=np.arange(len(class_names))
        )
    except ValueError:
        metrics["roc_auc"] = None
        print("ROC-AUC could not be calculated.")

    # Print evaluation results
    print("\n========== MODEL EVALUATION ==========")

    for metric, value in metrics.items():

        if value is not None:
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: Not available")

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0
    )

    print("\nCLASSIFICATION REPORT")
    print(report)

    with open(output_dir / "classification_report.txt", "w") as file:
        file.write(report)

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names))
    )

    fig, ax = plt.subplots(figsize=(11, 9))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=True
    )

    ax.set_title("DermaBench: Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        output_dir / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nEvaluation completed successfully!")
    print(f"Results saved to: {output_dir}")

    return metrics, y_true, y_pred, y_prob


if __name__ == "__main__":

    print("DermaBench Evaluator imported successfully!")
    print("Functions available:")
    print("- evaluate_model()")

