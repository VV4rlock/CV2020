from config import *

logger = logging.getLogger(__name__)
IMAGE_PATH = r"chess.jpg"
K = 0.05

W = np.ones((5, 5))

assert W.shape[0] % 2 == 1 and W.shape[1] % 2 == 1
W_y_offset = (W.shape[0] - 1) // 2
W_x_offset = (W.shape[1] - 1) // 2

y_kernel = (np.array([-1, 8, 0, -8, 1])/12).reshape(-1, 1)
x_kernel = y_kernel.T
logger.info("x:\n{}\ny:{}".format(x_kernel, y_kernel))


def harris_corner_detection(grey_image: np.ndarray):
    grey_image = grey_image.astype(np.float64)
    Ix = apply_filter(grey_image, x_kernel)
    Iy = apply_filter(grey_image, y_kernel)
    response = np.zeros(grey_image.shape, dtype=grey_image.dtype)
    temp = np.zeros(3, dtype=np.float64)
    for i in range(W_y_offset, grey_image.shape[0] - W_y_offset):
        for j in range(W_x_offset, grey_image.shape[1] - W_x_offset):
            temp *= 0
            for wi in range(W.shape[0]):
                for wj in range(W.shape[1]):
                    temp += W[wi, wj] * np.array([Ix[i + wi - W_y_offset, j + wj - W_x_offset] ** 2,
                                                  Ix[i + wi - W_y_offset, j + wj - W_x_offset] \
                                                  * Iy[i + wi - W_y_offset, j + wj - W_x_offset],
                                                  Iy[i + wi - W_y_offset, j + wj - W_x_offset] ** 2])

            response[i, j] = temp[0]*temp[2] - temp[1] ** 2 - K * (temp[0] + temp[2])
    #response = response[1:-1,1:-1]
    logger.info("response max={} min={}".format(response.max(), response.min()))
    show_image(response)
    response = response / response.max()
    grey_image = grey_image / grey_image.max()
    #k = 1
    #show_image(grey_image*k + response*(1 - k))






if __name__ == "__main__":
    img = cv2.imread(IMAGE_PATH)
    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    harris_corner_detection(grey_image)