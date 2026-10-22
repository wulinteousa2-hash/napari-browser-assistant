"""Explicit, versioned region encoders. No automatic model downloads."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

PREPROCESS = "gray-percentile-1-99-resize224-v1"


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gray_image(data):
    data = np.asarray(data)
    if data.ndim == 3:
        data = data[..., :3].astype(np.float32) @ np.array([.299, .587, .114], dtype=np.float32)
    if data.ndim != 2 or not data.size:
        raise ValueError("Encoder input must be a nonempty XY or RGB plane")
    data = data.astype(np.float32)
    finite = data[np.isfinite(data)]
    if not finite.size:
        raise ValueError("Region contains no finite pixels")
    low, high = np.percentile(finite, (1, 99))
    data = np.clip(np.nan_to_num((data - low) / max(float(high - low), 1e-6)), 0, 1)
    return np.asarray(Image.fromarray(data).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)


def unit_vector(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if not vector.size or not np.isfinite(vector).all():
        raise ValueError("Encoder returned an empty or non-finite embedding")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Encoder returned a zero embedding")
    return vector / norm


class Encoder:
    def __init__(self, kind="texture", weights="", device="cpu"):
        self.kind, self.weights, self.device = kind, str(weights), device
        self.model = None
        self.metadata = {"kind": kind, "preprocessing": PREPROCESS, "device": device}
        if kind == "texture":
            self.metadata.update(version="texture-grid-v1", learned=False)
        else:
            if not Path(weights).is_file():
                raise ValueError("Choose a local encoder checkpoint first")
            self.metadata["weights_sha256"] = file_digest(weights)
            import torch
            self.torch = torch
            if device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable; select CPU")
            self.metadata["torch_version"] = torch.__version__
            if kind == "resnet18":
                import torchvision
                self.metadata["backend_version"] = torchvision.__version__
                self.model = torchvision.models.resnet18(weights=None)
                if str(weights).lower().endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state = load_file(str(weights))
                else:
                    state = torch.load(str(weights), map_location="cpu", weights_only=True)
                    state = state.get("state_dict", state)
                state = {k: v for k, v in state.items() if not k.startswith("fc.")}
                self.model.fc = torch.nn.Identity()
                self.model.load_state_dict(state, strict=True)
                self.model = self.model.eval().to(device)
                self.metadata.update(version="resnet18-avgpool-v1", learned=True,
                                     normalization="ImageNet mean/std")
            elif kind == "emdino":
                try:
                    from omniem import EMEncoder
                except ImportError as error:
                    raise ValueError("Install omniem in this napari environment to use EM-DINO") from error
                from importlib.metadata import version
                self.metadata["backend_version"] = version("omniem")
                self.model = EMEncoder.load("emdinov1", str(weights)).eval().to(device)
                self.metadata.update(version="emdinov1-cls-v1", learned=True,
                                     normalization="omniem emdinov1 default")
            else:
                raise ValueError(f"Unknown encoder: {kind}")
        self.key = hashlib.sha256(__import__("json").dumps(self.metadata, sort_keys=True).encode()).hexdigest()

    def encode(self, data):
        gray = gray_image(data)
        if self.kind == "texture":
            blocks = gray.reshape(8, 28, 8, 28).transpose(0, 2, 1, 3)
            gy, gx = np.gradient(gray)
            histogram = np.histogram(gray, bins=16, range=(0, 1), density=False)[0] / gray.size
            features = np.concatenate(([1.0], blocks.mean((2, 3)).ravel(),
                                       blocks.std((2, 3)).ravel(), histogram,
                                       np.histogram(np.hypot(gx, gy), bins=8, range=(0, 1))[0] / gray.size))
            return unit_vector(features)
        with self.torch.inference_mode():
            tensor = self.torch.from_numpy(gray.copy()).to(self.device)
            if self.kind == "emdino":
                result = self.model.run(tensor, axes="yx", return_cls=True, return_patch=False)["cls"]
            else:
                tensor = tensor[None, None].repeat(1, 3, 1, 1)
                mean = tensor.new_tensor([.485, .456, .406])[None, :, None, None]
                std = tensor.new_tensor([.229, .224, .225])[None, :, None, None]
                result = self.model((tensor - mean) / std)
            return unit_vector(result.detach().float().cpu().numpy())
