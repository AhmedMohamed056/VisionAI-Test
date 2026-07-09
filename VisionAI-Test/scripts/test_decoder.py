import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ENCODER_PATH = ROOT / "models" / "mobilesam" / "encoder.onnx"
DECODER_PATH = ROOT / "models" / "mobilesam" / "decoder.onnx"

IMAGE_PATH = ROOT / "images" / "person.jpg"

# ------------------------
# Load image
# ------------------------

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

original = image.copy()

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

h, w = rgb.shape[:2]

resized = cv2.resize(rgb, (1024, 1024))

input_tensor = resized.astype(np.float32) / 255.0
input_tensor = np.transpose(input_tensor, (2, 0, 1))
input_tensor = np.expand_dims(input_tensor, axis=0)

# ------------------------
# Load models
# ------------------------

encoder = ort.InferenceSession(
    str(ENCODER_PATH),
    providers=["CPUExecutionProvider"],
)

decoder = ort.InferenceSession(
    str(DECODER_PATH),
    providers=["CPUExecutionProvider"],
)

print("Running encoder...")

embedding = encoder.run(
    None,
    {"image": input_tensor},
)[0]

print("Encoder Done")

clicked_point = None


def mouse(event, x, y, flags, param):
    global clicked_point

    if event == cv2.EVENT_LBUTTONDOWN:

        clicked_point = (x, y)


cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse)

while True:

    display = original.copy()

    if clicked_point:

        cv2.circle(display, clicked_point, 6, (0, 0, 255), -1)

    cv2.imshow("Image", display)

    key = cv2.waitKey(1)

    if key == 27:
        break

    if clicked_point is None:
        continue

    x, y = clicked_point

    sx = x * 1024 / w
    sy = y * 1024 / h

    point_coords = np.array(
        [[[sx, sy]]],
        dtype=np.float32,
    )

    point_labels = np.array(
        [[1]],
        dtype=np.float32,
    )

    masks, scores = decoder.run(
        None,
        {
            "image_embeddings": embedding,
            "point_coords": point_coords,
            "point_labels": point_labels,
        },
    )

    mask = masks[0, 0]

    mask = cv2.resize(mask, (w, h))

    mask = (mask > 0).astype(np.uint8) * 255

    cv2.imshow("Mask", mask)

    clicked_point = None

cv2.destroyAllWindows()