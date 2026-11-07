import torch
from pathlib import Path


CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    model,
    optimizer,
    epoch,
    accuracy,
    loss,
    filename="best_model.pth"
):

    checkpoint_path = CHECKPOINT_DIR / filename

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "accuracy": accuracy,
        "loss": loss
    }

    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}")

    return checkpoint_path


def load_checkpoint(
    model,
    optimizer=None,
    filename="best_model.pth",
    device="cpu"
):

    checkpoint_path = CHECKPOINT_DIR / filename

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    print(f"Checkpoint loaded: {checkpoint_path}")

    return checkpoint


if __name__ == "__main__":

    print("Checkpoint module imported successfully!")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print("Functions available:")
    print("- save_checkpoint()")
    print("- load_checkpoint()")
