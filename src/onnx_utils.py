"""Registry of inference models: how to rebuild each one from its checkpoint for ONNX export/verify.

Each entry: checkpoint file, a loader that returns the PyTorch model in eval mode on CPU, and the
input/output names used in the ONNX graph (the backend uses the same names).
Later phases add the classifier, specialists, soft-MoE and the GAN generator here.
"""
import functools

import torch

from src import config as C
from src.models.autoencoder import ConvAutoencoder
from src.models.classifier import CorruptionClassifier


def load_autoencoder(path) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ConvAutoencoder(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()  # eval(): dropout off, BatchNorm uses its running statistics


class ClassifierWithProbs(torch.nn.Module):
    """Export wrapper: returns logits (needed by the Task 3 gate) AND softmax probabilities (shown in the app)."""

    def __init__(self, classifier: torch.nn.Module):
        super().__init__()
        self.classifier = classifier

    def forward(self, x):
        logits = self.classifier(x)
        return logits, torch.softmax(logits, dim=1)


def load_classifier(path, with_probs: bool = False) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CorruptionClassifier(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return ClassifierWithProbs(model).eval() if with_probs else model


REGISTRY = {
    "task1_universal_ae": {
        "checkpoint": C.CHECKPOINTS / "task1_universal_ae.pt",
        "load": load_autoencoder,
        "inputs": ["input"],      # float32 [N, 3, 128, 128] in [0, 1]
        "outputs": ["output"],    # float32 [N, 3, 128, 128] in [0, 1]
    },
    "task2_classifier": {
        "checkpoint": C.CHECKPOINTS / "task2_classifier.pt",
        "load": functools.partial(load_classifier, with_probs=True),
        "inputs": ["input"],
        "outputs": ["logits", "probs"],   # [N, 4] each, order: clean, salt, blur, occlusion
    },
}
for _name in ("salt", "blur", "occlusion"):
    REGISTRY[f"task2_expert_{_name}"] = {
        "checkpoint": C.CHECKPOINTS / f"task2_expert_{_name}.pt",
        "load": load_autoencoder,
        "inputs": ["input"],
        "outputs": ["output"],
    }
