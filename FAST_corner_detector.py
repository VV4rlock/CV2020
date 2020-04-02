from config import *

scales = [1, 1/2, 1/4, 1/8]

def FAST(grey_image: np.ndarray, R=9, T=100, offset=0) -> np.ndarray:
    H, W = grey_image.shape

    offset = max(R, offset)
    grey_image = grey_image.astype(np.int32)

    fast_1 = grey_image[offset - R: H - offset - R, offset: W - offset]
    fast_5 = grey_image[offset: H - offset, offset + R: W - offset + R]
    fast_9 = grey_image[offset + R: H - offset + R, offset: W - offset]
    fast_13 = grey_image[offset: H - offset, offset - R: W - offset - R]
    Ip_arr = grey_image[offset:H - offset, offset:W - offset]

    greater = (Ip_arr - fast_1 > T).astype(np.int8) + \
           (Ip_arr - fast_5 > T) + \
           (Ip_arr - fast_9 > T) + \
           (Ip_arr - fast_13 > T)

    less = (fast_1 - Ip_arr > T).astype(np.int8) + \
              (fast_5 - Ip_arr > T) + \
              (fast_9 - Ip_arr > T) + \
              (fast_13 - Ip_arr > T)

    Y, X = np.mgrid[-R:R + 1, -R:R + 1]
    mask = (X ** 2 + Y ** 2 <= R ** 2 + 1) * ((R - 1) ** 2 + 2 < X ** 2 + Y ** 2)
    keep_len = 0
    route = [[], [], [], []]
    for i, j in ((y, x) for x in range(R, 0, -1) for y in range(R + 1)):
        if mask[R + i, R + j]:
            keep_len += 1
            route[0].append((-j, i))
            route[1].append((i, j))
            route[2].append((j, -i))
            route[3].append((-i, -j))
    route = sum(route, [])
    route = np.array(route) + R
    t_route = (route[:, 0], route[:, 1])
    keep_len *= 3

    res, small_W, patch_size = [], W - 2 * offset, 2 * R + 1
    for i in np.sort(np.unique(np.concatenate((np.where((less > 2).reshape(-1))[0],
                                               np.where((greater > 2).reshape(-1))[0])))):
        x, y = i // small_W, i % small_W

        temp = (np.abs(Ip_arr[x, y] - grey_image[x: x + patch_size, y: y + patch_size][t_route]) > T).astype(np.int8)

        prev, zero_indexes, sums = 0, np.where(temp == 0)[0], []
        if len(zero_indexes) < 2:
            res.append((x + offset, y + offset))
            continue
        for zero_index in zero_indexes:
            sums.append(temp[prev:zero_index].sum())
            prev = zero_index
        if temp[0] == 1 == temp[1]:
            sums[0] += sums[-1]
        if max(sums) >= keep_len:
            res.append((x + offset, y + offset))
    return np.array(res)


def oriented_FAST(grey_image: np.ndarray, R=9, fast_radius=9, fast_threshold=100, offset=0):
    ORIENTED_R = R
    grey_image = grey_image.astype(np.int32)
    angles = FAST(grey_image, R=fast_radius, T=fast_threshold, offset=offset)
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



