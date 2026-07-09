from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class MaskConfig:
    """
    Configuration for mask post-processing.
    """

    threshold: float = 0.5
    kernel_size: int = 5
    feather_radius: int = 7
    keep_largest: bool = True


class MaskProcessor:
    """
    Cleans MobileSAM masks before sending them to LaMa.
    """

    def __init__(self, config: MaskConfig | None = None):
        self.config = config or MaskConfig()

    def threshold(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert probability mask into binary uint8 mask.
        """

        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)

        binary = (mask >= self.config.threshold).astype(np.uint8)
        return binary * 255

    def keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """
        Keep only the largest connected component.
        """

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        if num_labels <= 1:
            return mask

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        output = np.zeros_like(mask)
        output[labels == largest] = 255

        return output

    def remove_noise(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove tiny blobs.
        """

        kernel = np.ones(
            (
                self.config.kernel_size,
                self.config.kernel_size,
            ),
            np.uint8,
        )

        return cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

    def closing(self, mask: np.ndarray) -> np.ndarray:
        """
        Fill small holes.
        """

        kernel = np.ones(
            (
                self.config.kernel_size,
                self.config.kernel_size,
            ),
            np.uint8,
        )

        return cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

    def dilate(self, mask: np.ndarray) -> np.ndarray:
        """
        Expand mask slightly.
        """

        kernel = np.ones(
            (
                self.config.kernel_size,
                self.config.kernel_size,
            ),
            np.uint8,
        )

        return cv2.dilate(
            mask,
            kernel,
            iterations=1,
        )

    def feather(self, mask: np.ndarray) -> np.ndarray:
        """
        Blur mask edges for smoother inpainting.
        """

        radius = self.config.feather_radius

        if radius <= 0:
            return mask

        k = radius * 2 + 1

        return cv2.GaussianBlur(
            mask,
            (k, k),
            0,
        )

    def process(self, mask: np.ndarray) -> np.ndarray:
        """
        Complete post-processing pipeline.
        """

        mask = self.threshold(mask)

        if self.config.keep_largest:
            mask = self.keep_largest_component(mask)

        mask = self.remove_noise(mask)
        mask = self.closing(mask)
        mask = self.dilate(mask)
        mask = self.feather(mask)

        return mask