import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "mobilesam" / "encoder.onnx"
IMAGE_PATH = ROOT / "images" / "person.jpg"


def preprocess(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_size = image.shape[:2]

    image = cv2.resize(image, (1024, 1024))

    image = image.astype(np.float32) / 255.0

    image = np.transpose(image, (2, 0, 1))

    image = np.expand_dims(image, axis=0)

    return image, original_size


def main():

    print("=" * 60)
    print("Loading MobileSAM Encoder...")
    print("=" * 60)

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )

    print("\nModel Loaded Successfully\n")

    print("Inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}")
        print(f"    shape : {inp.shape}")
        print(f"    dtype : {inp.type}")

    print("\nOutputs:")
    for out in session.get_outputs():
        print(f"  {out.name}")
        print(f"    shape : {out.shape}")
        print(f"    dtype : {out.type}")

    image, original_size = preprocess(IMAGE_PATH)

    print("\nOriginal Size :", original_size)
    print("Input Tensor :", image.shape)

    start = time.time()

    embedding = session.run(
        None,
        {"image": image}
    )[0]

    end = time.time()

    print("\nEmbedding Shape :", embedding.shape)

    print("Embedding dtype :", embedding.dtype)

    print(f"Inference Time : {(end-start)*1000:.2f} ms")


if __name__ == "__main__":
    main()