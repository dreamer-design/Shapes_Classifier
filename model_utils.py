import torch
import time
import os

# -------------------------------
# 🧱 Basic Model Save/Load Utils
# -------------------------------

def save_model_state(model, path="model.pth"):
    """Save only model weights."""
    torch.save(model.state_dict(), path)
    print(f"✅ Model weights saved to {path}")


def load_model_state(model, path, device="cpu"):
    """Load model weights into an existing model."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    print(f"✅ Model weights loaded from {path}")
    return model


# -------------------------------
# 💾 Checkpoint Save/Load Utils
# -------------------------------

def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pth"):
    """Save model + optimizer + training state."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    torch.save(checkpoint, path)
    print(f"💾 Checkpoint saved at epoch {epoch} → {path}")


def load_checkpoint(model, optimizer, path, device="cpu"):
    """Resume training from a checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"✅ Resumed from epoch {epoch}, loss={loss:.4f}")
    return model, optimizer, epoch, loss


# -------------------------------
# 🕒 Utility Helpers
# -------------------------------

def save_model_with_timestamp(model, prefix="model"):
    """Save model with a timestamped filename."""
    timestamp = int(time.time())
    fname = f"{prefix}_{timestamp}.pth"
    save_model_state(model, fname)
    return fname


# -------------------------------
# 🌍 ONNX Export
# -------------------------------

def export_to_onnx(model, input_shape=(1, 3, 64, 64), path="model.onnx"):
    """Export model to ONNX format for deployment."""
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(model, dummy_input, path, opset_version=17)
    print(f"🌍 Exported model to {path}")
