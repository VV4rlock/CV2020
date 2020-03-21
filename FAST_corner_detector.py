from config import *

scales = [1, 1/2, 1/4, 1/8]

def FAST(grey_image: np.ndarray, R=9, t=100, offset=0) -> (list, list):
    H, W = grey_image.shape

    T = t
    FAST_R = R
    R = max(FAST_R, offset)
    grey_image = grey_image.astype(np.int32)

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
    route = np.array(route) + FAST_R
    t_route = (route[:, 0], route[:, 1])
    keep_len *= 3

    res, small_W, patch_size = [], W - 2 * R, 2 * FAST_R + 1
    for i in np.sort(np.unique(np.concatenate((np.where((less > 2).reshape(-1))[0],
                                               np.where((greater > 2).reshape(-1))[0])))):
        x, y = i // small_W, i % small_W

        temp = (np.abs(Ip_arr[x, y] - grey_image[x: x + patch_size, y: y + patch_size][t_route]) > T).astype(np.int8)

        prev, zero_indexes, sums = 0, np.where(temp == 0)[0], []
        if len(zero_indexes) < 2:
            res.append((x + R, y + R))
            continue
        for zero_index in zero_indexes:
            sums.append(temp[prev:zero_index].sum())
            prev = zero_index
        if temp[0] == 1 == temp[1]:
            sums[0] += sums[-1]
        if max(sums) >= keep_len:
            res.append((x + R, y + R))

    return np.array(res)


def oriented_FAST(grey_image: np.ndarray, R=9, fast_radius=9, fast_threshold=100, offset=0):
    ORIENTED_R = R
    grey_image = grey_image.astype(np.int32)
    angles = FAST(grey_image, R=fast_radius, t=fast_threshold, offset=offset)
    Y, X = np.mgrid[-ORIENTED_R:ORIENTED_R + 1, -ORIENTED_R:ORIENTED_R + 1]
    mask = (X ** 2 + Y ** 2 <= ORIENTED_R ** 2 + 1).astype(np.int16)

    m01 = np.zeros(len(angles), dtype=np.int32)
    m10 = np.zeros(len(angles), dtype=np.int32)

    for index, coord in enumerate(angles):
        part = grey_image[coord[0]-ORIENTED_R:coord[0]+ORIENTED_R + 1, coord[1] - ORIENTED_R: coord[1] + ORIENTED_R + 1]
        m01[index] = (mask * part * Y).sum()
        m10[index] = (mask * part * X).sum()

    theta = np.arctan2(m01, m10)

    return angles, theta


if __name__ == "__main__":
    img_input = cv2.imread(IMAGE_PATH)
    print(img_input)
    res = {}
    for scale in scales:
        img = cv2.resize(img_input.copy(), (int(img_input.shape[1] * scale), int(img_input.shape[0] * scale)))
        grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
        res[scale] = oriented_FAST(grey_image)

    print(res)
    #ales_less, angles_greaterng = FAST(grey_image)



