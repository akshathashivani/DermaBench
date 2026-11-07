import json

import torch

from src.data.load_data import create_dataloaders
from src.data.preprocessing import calculate_class_weights

from src.models.resnet import DermaResNet18
from src.models.efficientnet import DermaEfficientNetB0
from src.models.vit import DermaViT

from src.training.trainer import train_model
from src.training.config import DEVICE, OUTPUT_DIR, RANDOM_SEED


def run_experiments():

    torch.manual_seed(RANDOM_SEED)

    print("\n========== DERMABENCH EXPERIMENTS ==========")
    print("Device:", DEVICE)

    train_loader, val_loader, test_loader = create_dataloaders()

    class_weights = calculate_class_weights(
        train_loader.dataset.labels
    )

    models = {
        "ResNet18": DermaResNet18,
        "EfficientNetB0": DermaEfficientNetB0,
        "ViT": DermaViT
    }

    results = {}

    for model_name, model_class in models.items():

        print(f"\n========== {model_name} ==========")

        model = model_class(num_classes=7)

        trained_model, history = train_model(
            model,
            train_loader,
            val_loader,
            class_weights=class_weights,
            model_name=model_name.lower()
        )

        results[model_name] = {
            "best_validation_accuracy": max(
                history["val_accuracy"]
            ),
            "best_training_accuracy": max(
                history["train_accuracy"]
            ),
            "final_validation_loss": history["val_loss"][-1]
        }

    results_path = OUTPUT_DIR / "experiment_results.json"

    with open(results_path, "w") as file:
        json.dump(results, file, indent=4)

    print("\n========== EXPERIMENT SUMMARY ==========")

    for name, metrics in results.items():
        print(
            f"{name}: "
            f"Validation Accuracy = "
            f"{metrics['best_validation_accuracy'] * 100:.2f}%"
        )

    print("\nResults saved to:", results_path)


if __name__ == "__main__":
    run_experiments()
