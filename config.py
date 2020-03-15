import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging

logging.basicConfig(format="[%(filename)s: %(funcName)s] %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)
SMALL_SCREEN = False

IMAGE_PATH = r"shape.png"
N = 2

ONES = np.ones((2 * N + 1, 2 * N + 1), dtype=np.float32)
GAUSSIAN = np.array([  [ 1,  4,  7,  4,  1],
                       [ 4, 16, 26, 16,  4],
                       [ 7, 26, 41, 26,  7],
                       [ 4, 16, 26, 16,  4],
                       [ 1,  4,  7,  4,  1]], dtype=np.float32)

LUCAS_CANADE = (np.array([-1, 8, 0, -8, 1]) / 12).reshape(-1, 1)
PREWITT = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]])
SOBEL = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])


def show_image(img, name='image'):
    #cv2.destroyAllWindows()
    max, min = img.max(), img.min()
    img = ((img - min) / (max-min) * 255).astype(np.uint8)

    if SMALL_SCREEN:
        img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def apply_filter(image, filter) -> np.ndarray:
    return cv2.filter2D(image, -1, filter)