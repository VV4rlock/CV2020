from config import *

T = 120
FAST_R = 3
ORIENTED_R = 9
scales = [1, 1/2, 1/4, 1/8]

def FAST(grey_image: np.ndarray) -> (list, list):
    H, W = grey_image.shape

    R = max(FAST_R, ORIENTED_R)
    fast_1 = grey_image[0: H - 2 * R, R: W - R]
    fast_5 = grey_image[R: H - R, 2 * R: W]
    fast_9 = grey_image[2 * R: H, R: W - R]
    fast_13 = grey_image[R: H - R, 0: W - 2 * R]
    Ip_arr = grey_image[R:H - R, R:W - R]

    less = (Ip_arr - fast_1 > T).astype(np.int8) + \
           (Ip_arr - fast_5 > T) + \
           (Ip_arr - fast_9 > T) + \
           (Ip_arr - fast_13 > T)

    greater = (fast_1 - Ip_arr > T).astype(np.int8) + \
              (fast_5 - Ip_arr > T) + \
              (fast_9 - Ip_arr > T) + \
              (fast_13 - Ip_arr > T)

    Y, X = np.mgrid[-FAST_R:FAST_R + 1, -FAST_R:FAST_R + 1]
    mask = (X ** 2 + Y ** 2 <= FAST_R ** 2 + 1) * ((FAST_R - 1) ** 2 + 2 < X ** 2 + Y ** 2)
    keep_len = 0
    route = [[], [], [], []]
    for i, j in ((y, x) for x in range(FAST_R, 0, -1) for y in range(FAST_R + 1)):
        if mask[FAST_R + i, FAST_R + j]:
            keep_len += 1
            route[0].append((-j, i))
            route[1].append((i, j))
            route[2].append((j, -i))
            route[3].append((-i, -j))
    route = sum(route, [])
    keep_len *= 3

    res_less = []
    for i in np.squeeze(np.where((less > 2).reshape(-1))):
        x, y = i // (W - 2 * R), i % (W - 2 * R)

        count, first_seq, last_check = 0, None, True
        temp = Ip_arr[x, y] - grey_image[x: x + 2 * FAST_R + 1, y: y + 2 * FAST_R + 1] * mask > T

        if temp.sum() < keep_len:
            continue

        for ix, iy in route:
            if (temp[ix + FAST_R, iy + FAST_R] == 1):
                count += 1
            else:
                if first_seq is None:
                    first_seq = count
                count = 0

            if (count == keep_len):
                res_less.append((x + FAST_R, y + FAST_R))
                last_check = False
                break
        if last_check and count + first_seq >= keep_len:
            res_less.append((x + FAST_R, y + FAST_R))

    res_greater = []
    for i in np.squeeze(np.where((greater > 2).reshape(-1))):
        x, y = i // (W - 2 * R), i % (W - 2 * R)
        count, first_seq, last_check = 0, None, True
        temp = grey_image[x: x + 2 * FAST_R + 1, y: y + 2 * FAST_R + 1] * mask - Ip_arr[x, y] > T
        if temp.sum() < keep_len:
            continue
        for ix, iy in route:
            if (temp[ix + FAST_R, iy + FAST_R] == 1):
                count += 1
            else:
                if first_seq is None:
                    first_seq = count
                count = 0

            if ( count == keep_len ):
                res_greater.append((x + FAST_R, y + FAST_R))
                last_check = False
                break
        if last_check and count + first_seq >= keep_len:
            res_greater.append((x + FAST_R, y + FAST_R))

    return res_less, res_greater


def oriented_FAST(grey_image: np.ndarray, original_image=None):
    angles_less, angles_greater = FAST(grey_image)
    Y, X = np.mgrid[-ORIENTED_R:ORIENTED_R + 1, -ORIENTED_R:ORIENTED_R + 1]
    mask = (X ** 2 + Y ** 2 <= ORIENTED_R ** 2 + 1).astype(np.int16)


    print(len(angles_less))
    m01 = np.zeros(len(angles_less), dtype=np.int32)
    m10 = np.zeros(len(angles_less), dtype=np.int32)
    color = (0, 0, 255)
    line_c = (0, 255, 0)
    for index, coord in enumerate(angles_less):
        part = grey_image[coord[0]-ORIENTED_R:coord[0]+ORIENTED_R + 1, coord[1] - ORIENTED_R: coord[1] + ORIENTED_R + 1]
        m01[index] = (mask * part * Y).sum()
        m10[index] = (mask * part * X).sum()

        if original_image is not None:
            l = 10 / (m01[index] ** 2 + m10[index] ** 2) ** 0.5
            cv2.circle(original_image, (coord[1], coord[0]), 10, color, 1)
            cv2.line(original_image,
                     (coord[1],coord[0]), (coord[1]+int(m10[index]*l),coord[0] + int(m01[index]*l)), line_c, 1)

    theta_less = np.arctan2(m01, m10)

    m01 = np.zeros(len(angles_greater), dtype=np.int32)
    m10 = np.zeros(len(angles_greater), dtype=np.int32)
    color = (0, 255, 255)
    for index, coord in enumerate(angles_greater):
        part = grey_image[coord[0]-ORIENTED_R:coord[0]+ORIENTED_R + 1, coord[1] - ORIENTED_R: coord[1] + ORIENTED_R + 1]
        m01[index] = (mask * part * Y).sum()
        m10[index] = (mask * part * X).sum()

        if original_image is not None:
            l = 10 / (m01[index] ** 2 + m10[index] ** 2) ** 0.5
            cv2.circle(original_image, (coord[1], coord[0]), 10, color, 1)
            cv2.line(original_image,
                     (coord[1], coord[0]), (coord[1] + int(m10[index] * l), coord[0] + int(m01[index] * l)), line_c, 1)
    theta_greater = np.arctan2(m01, m10)

    if original_image is not None:
        show_image(original_image)

    return ((angles_less, theta_less), (angles_greater, theta_greater))


if __name__ == "__main__":
    img_input = cv2.imread(IMAGE_PATH)
    res = {}
    for scale in scales:
        img = cv2.resize(img_input.copy(), (int(img_input.shape[1] * scale), int(img_input.shape[0] * scale)))
        grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
        res[scale] = oriented_FAST(grey_image, img)

    print(res)
    #angles_less, angles_greater = FAST(grey_image)



