from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms


def main():
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = datasets.MNIST(
        root="data/mnist",
        train=False,
        download=True,
        transform=transform
    )

    num_samples = 10000

    images = []
    labels = []

    for i in range(num_samples):
        image, label = test_dataset[i]

        # image shape is [1, 28, 28], values already in [0, 1]
        images.append(image.numpy().astype(np.float32))
        labels.append(label)

    images = np.stack(images, axis=0).astype(np.float32)
    labels = np.array(labels, dtype=np.uint8)

    images.tofile(out_dir / "test_images.bin")
    labels.tofile(out_dir / "test_labels.bin")

    print("Wrote data/test_images.bin")
    print("Wrote data/test_labels.bin")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("First 10 labels:", labels[:10])


if __name__ == "__main__":
    main()