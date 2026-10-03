"""Registry of inference models: how to rebuild each one from its checkpoint for ONNX export/verify.

Each entry: checkpoint file, a loader that returns the PyTorch model in eval mode on CPU, and the
input/output names used in the ONNX graph (the backend uses the same names).
Later phases add the classifier, specialists, soft-MoE and the GAN generator here.
"""
import torch

from src import config as C
from src.models.autoencoder import ConvAutoencoder


def load_autoencoder(path) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ConvAutoencoder(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()  # eval(): dropout off, BatchNorm uses its running statistics


REGISTRY = {
    "task1_universal_ae": {
        "checkpoint": C.CHECKPOINTS / "task1_universal_ae.pt",
        "load": load_autoencoder,
        "inputs": ["input"],      # float32 [N, 3, 128, 128] in [0, 1]
        "outputs": ["output"],    # float32 [N, 3, 128, 128] in [0, 1]
    },
}
