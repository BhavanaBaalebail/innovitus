#!/usr/bin/env python3
"""Export best_model.pth → fatigue_model.onnx (opset 18). Architecture must match train.py."""

import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


class FatigueDetectionCNN(nn.Module):
    """Same as train.py FatigueDetectionCNN (Conv-BN-ReLU + classifier)."""

    def __init__(self, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    device = torch.device("cpu")
    model = FatigueDetectionCNN(dropout=0.25).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
    model.eval()

    dummy = torch.randn(1, 3, 64, 64)
    onnx_path = "fatigue_model.onnx"
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["image"],
        output_names=["output"],
        opset_version=18,
        dynamic_axes={"image": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Exported {onnx_path} (opset 18)")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    out = session.run(None, {name: dummy.numpy()})[0]
    ref = model(dummy).detach().numpy()
    if np.allclose(out, ref, atol=1e-2):
        print("ONNX vs PyTorch: OK")
    else:
        print("ONNX vs PyTorch: check numeric diff (BN may differ slightly)")

    mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Size: {mb:.2f} MB")


if __name__ == "__main__":
    main()
