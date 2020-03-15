from config import *

logger = logging.getLogger(__name__)
K = 0.05
T = 100


def shi_tomasi_response(a, b, c):
    return (a + c - (4*b*b + (a-c)**2) ** 0.5) / 2


def harris_response(a, b, c) -> np.ndarray:
    return a * c - b * b - K * (a + c) * (a + c)

RESPONSE_FUNC = harris_response

W = ONES
W = W / W.sum()

y_kernel = PREWITT
x_kernel = y_kernel.T
#logger.info("x:\n{}\ny:{}".format(x_kernel, y_kernel))


def normalize(image: np.ndarray) -> np.ndarray:
    image = (image - image.min())
    return image / image.max()


def mirrirPad(image: np.ndarray, h, w) -> np.ndarray:
    out = np.zeros((image.shape[0] + 2 * h, image.shape[1] + 2 * w), dtype=image.dtype)
    H, W = out.shape
    out[h:H - h, w:W-w] = image

    out[0:h, 0:w] = np.flip(np.flip(out[h:2 * h, w: 2 * w], 0), 1)
    out[0:h, W-w:W] = np.flip(np.flip(out[h:2 * h, W - 2 * w: W - w], 0), 1)
    out[H-h:H, W - w:W] = np.flip(np.flip(out[H-2*h:H - h, W - 2 * w: W - w], 0), 1)
    out[H - h:H, 0:w] = np.flip(np.flip(out[H - 2 * h:H - h, w: 2 * w], 0), 1)

    out[0:h, w:W-w] = np.flip(out[h:2 * h, w:W-w], 0)
    out[H-h:H, w:W - w] = np.flip(out[H-2*h:H - h, w:W - w], 0)
    out[h:H-h, 0:w] = np.flip(out[h:H-h, w: 2 * w], 1)
    out[h:H - h, W-w:W] = np.flip(out[h:H - h, W - 2*w: W - w], 1)
    return out


def NMS(image: np.ndarray) -> np.ndarray:
    n = 2
    H, W = image.shape
    for i, j in ((x, y) for x in range(H) for y in range(W)):
        if image[i, j] != image[max(0, i - n): min(i + n, W), max(0, j - n): min(j + n, W)].max():
            image[i, j] = 0
    return image


def filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    H,W = image.shape
    h,w = kernel.shape
    img = mirrirPad(image, H + h - 1, W + w - 1)
    res = np.zeros((H, W), dtype=np.float32)
    for i, j in ((x, y) for x in range(h) for y in range(w)):
        res += kernel[h - 1 - i, w - 1 - j] * img[i:H + i, j: W + j]
    return res #np.rot90(res, 2)


def compare(img1, img2):
    im = np.zeros((img1.shape[0], img1.shape[1],3), dtype=np.uint8)
    img1 = np.abs(img1)
    img2 = np.abs(img2)
    img1=img1 / img1.max() * 255
    img2 = img2 / img2.max() * 255
    im[:, :, 1] = img1
    im[:, :, 2] = img2
    cv2.imshow('compare images', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def harris_response(grey_image):
    grey_image_work = grey_image.astype(np.float64)
    Ix = filter(grey_image_work, x_kernel)
    Iy = filter(grey_image_work, y_kernel)

    Sx2 = filter(Ix * Ix, W)
    Sy2 = filter(Iy * Iy, W)
    Sxy = filter(Ix * Iy, W)

    response = RESPONSE_FUNC(Sx2, Sxy, Sy2)
    response = (normalize(NMS(response)) * 255).astype(np.uint8)
    return response


def harris_corner_detection(img: np.ndarray):
    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    response = harris_response(grey_image)
    #show_image(response)
    response[response < T] = 0
    #hist, edges = np.histogram(response, range(257))
    #plt.hist(response)
    #plt.show()


    #show_image(response)
    cnts, _ = cv2.findContours(response, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        color = (0, 0, 255)#(np.random.rand(3) * 255).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(c)
        #cv2.rectangle(grey_image, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.circle(img, (x + w//2, y + h // 2), 10, color, 2)
    show_image(img)



if __name__ == "__main__":
    img = cv2.imread(IMAGE_PATH)
    harris_corner_detection(img)