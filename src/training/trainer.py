import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.training.engine import (
    train_one_epoch,
    validate_one_epoch
)

from src.training.checkpoint import save_checkpoint

from src.training.config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY
)


def train_model(
    model,
    train_loader,
    val_loader,
    class_weights=None,
    model_name="dermabench_model"
):

    print(f"\nStarting training: {model_name}")
    print(f"Device: {DEVICE}")

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
        if class_weights is not None else None
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_accuracy = 0.0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(NUM_EPOCHS):

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        val_loss, val_accuracy = validate_one_epoch(
            model, val_loader, criterion, DEVICE
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_accuracy,
                val_loss,
                filename=f"{model_name}_best.pth"
            )

            print("Best model saved!")

    print("\nTraining completed!")
    print(f"Best Validation Accuracy: {best_accuracy * 100:.2f}%")

    return model, history


if __name__ == "__main__":
    print("DermaBench Trainer imported successfully!")
    print("Functions available:")
    print("- train_model()")
