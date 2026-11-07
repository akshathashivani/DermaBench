import matplotlib.pyplot as plt

from pathlib import Path


def plot_training_history(
    history,
    model_name="DermaBench",
    output_dir="reports/figures"
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_keys = [
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy"
    ]

    for key in required_keys:
        if key not in history:
            raise ValueError(f"Missing history key: {key}")

    epochs = range(1, len(history["train_loss"]) + 1)

    if not history["train_loss"]:
        raise ValueError("Training history is empty.")

    if any(len(history[key]) != len(epochs) for key in required_keys):
        raise ValueError("Training history lengths do not match.")

    # Loss visualization
    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss",
        linewidth=2
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
        linewidth=2
    )

    plt.title(f"{model_name} - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{model_name}_loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Accuracy visualization
    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        [value * 100 for value in history["train_accuracy"]],
        label="Training Accuracy",
        linewidth=2
    )

    plt.plot(
        epochs,
        [value * 100 for value in history["val_accuracy"]],
        label="Validation Accuracy",
        linewidth=2
    )

    plt.title(f"{model_name} - Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{model_name}_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nTraining visualizations generated successfully!")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":

    print("DermaBench Visualizer imported successfully!")
    print("Functions available:")
    print("- plot_training_history()")
