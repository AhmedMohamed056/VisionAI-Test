import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

from vision.mask import MaskProcessor

ROOT = Path(__file__).resolve().parent.parent

ENCODER_PATH = ROOT / "models" / "mobilesam" / "encoder.onnx"
DECODER_PATH = ROOT / "models" / "mobilesam" / "decoder.onnx"

IMAGE_PATH = ROOT / "images" / "person.jpg"

OUTPUT_DIR = ROOT / "outputs" / "debug"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("VisionAI - MobileSAM Multi Point Test")
print("=" * 60)

# ============================================================
# Load Image
# ============================================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

original = image.copy()

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height, width = rgb.shape[:2]

resized = cv2.resize(rgb, (1024, 1024))

input_tensor = resized.astype(np.float32) / 255.0
input_tensor = np.transpose(input_tensor, (2, 0, 1))
input_tensor = np.expand_dims(input_tensor, axis=0)

# ============================================================
# Load Models
# ============================================================

print("Loading Encoder...")
encoder = ort.InferenceSession(
    str(ENCODER_PATH),
    providers=["CPUExecutionProvider"],
)

print("Loading Decoder...")
decoder = ort.InferenceSession(
    str(DECODER_PATH),
    providers=["CPUExecutionProvider"],
)

print("Running encoder once...")

embedding = encoder.run(
    None,
    {
        "image": input_tensor
    }
)[0]

print("Encoder Finished.")
print()

# ============================================================
# UI
# ============================================================

positive_points = []
negative_points = []

print("---------------------------------------------")
print("Controls")
print("---------------------------------------------")
print("Left Click  : Positive Point")
print("Right Click : Negative Point")
print("ENTER       : Run Segmentation")
print("C           : Clear Points")
print("ESC         : Exit")
print("---------------------------------------------")


def mouse_callback(event, x, y, flags, param):

    global positive_points
    global negative_points

    if event == cv2.EVENT_LBUTTONDOWN:
        positive_points.append((x, y))
        print(f"[+] Positive : ({x},{y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        negative_points.append((x, y))
        print(f"[-] Negative : ({x},{y})")


cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

processor = MaskProcessor()

while True:

    display = original.copy()

    # Draw positive points
    for p in positive_points:
        cv2.circle(display, p, 6, (0, 255, 0), -1)

    # Draw negative points
    for p in negative_points:
        cv2.circle(display, p, 6, (0, 0, 255), -1)

    cv2.imshow("Image", display)

    key = cv2.waitKey(1) & 0xFF

    # ESC
    if key == 27:
        break

    # Clear
    if key == ord("c"):
        positive_points.clear()
        negative_points.clear()
        print("Points Cleared.")

    # ENTER
    if key == 13:

        if len(positive_points) == 0:
            print("Please add at least one positive point.")
            continue

        coords = []
        labels = []

        # Positive
        for x, y in positive_points:

            sx = x * 1024 / width
            sy = y * 1024 / height

            coords.append([sx, sy])
            labels.append(1)

        # Negative
        for x, y in negative_points:

            sx = x * 1024 / width
            sy = y * 1024 / height

            coords.append([sx, sy])
            labels.append(0)

        point_coords = np.array(
            [coords],
            dtype=np.float32,
        )

        point_labels = np.array(
            [labels],
            dtype=np.float32,
        )

        print()
        print("=" * 50)
        print("Running Decoder")
        print("=" * 50)

        masks, scores = decoder.run(
            None,
            {
                "image_embeddings": embedding,
                "point_coords": point_coords,
                "point_labels": point_labels,
            },
        )

        raw_mask = masks[0, 0]

        raw_mask = cv2.resize(
            raw_mask,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        cv2.imwrite(
            str(OUTPUT_DIR / "01_raw_mask.png"),
            (raw_mask * 255).astype(np.uint8),
        )

        threshold = processor.threshold(raw_mask)

        cv2.imwrite(
            str(OUTPUT_DIR / "02_threshold.png"),
            threshold,
        )

        largest = processor.keep_largest_component(threshold)

        cv2.imwrite(
            str(OUTPUT_DIR / "03_largest_component.png"),
            largest,
        )

        overlay = original.copy()

        green = np.zeros_like(original)
        green[:, :] = (0, 255, 0)

        mask_bool = largest == 255

        overlay[mask_bool] = cv2.addWeighted(
            original,
            0.3,
            green,
            0.7,
            0,
        )[mask_bool]

        cv2.imwrite(
            str(OUTPUT_DIR / "04_overlay.png"),
            overlay,
        )

        print()
        print("Score :", float(scores[0][0]))
        print("Positive :", len(positive_points))
        print("Negative :", len(negative_points))
        print()
        print("Saved:")
        print("01_raw_mask.png")
        print("02_threshold.png")
        print("03_largest_component.png")
        print("04_overlay.png")
        print()

        cv2.imshow("Mask", largest)
        cv2.imshow("Overlay", overlay)

cv2.destroyAllWindows()