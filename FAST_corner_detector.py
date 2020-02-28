from config import *

T = 100

pixels = [(-3, 0), (-3, 1), (-2, 2), (-1, 3), (0, 3), (1, 3), (2, 2), (3, 1),
          (3, 0), (3, -1), (2, -2), (1, -3), (0, -3), (-1, -3), (-2, -2), (-3, -1)]
'''сори, этот вариант только для радиуса 3))) исправим для любого, а пока так. 
нужно генерировать массив pixels в порядек обхода. 
Вычислять константы n (количество последовательных пикелей) 
А также пока непонятно как из маски и изображения сгенерировать массив 16 пикселей в нужном порятке, а не в порядке 
    слева направо и сверху вниз) 

Этот вариант работает достаточно быстро, но можно попытаться улучшить с помощью вычисления суммы
    на изображении после наложения маски, отбрасывая сразу все, кроме 13 и 14 (для них уже делать полную проверку 
    на нахождение последовательности). Не знаю будет ли это намного эффективнее
'''

R = 3


def FAST(image: np.ndarray):
    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    H, W = grey_image.shape
    res = []

    fast_1 = grey_image[0: H - 2 * R, R: W - R]
    fast_5 = grey_image[R: H - R, 2 * R: W]
    fast_9 = grey_image[2 * R: H, R: W - R]
    fast_13 = grey_image[R: H - R, 0: W - 2 * R]
    Ip_arr = grey_image[R:H-R, R:W-R]

    #print(fast_1.shape, fast_5.shape, fast_9.shape, fast_13.shape, Ip_arr.shape)
    less = (Ip_arr - fast_1 > T).astype(np.int8) + \
           (Ip_arr - fast_5 > T) + \
           (Ip_arr - fast_9 > T) + \
           (Ip_arr - fast_13 > T)

    greater = (fast_1 - Ip_arr > T).astype(np.int8) + \
              (fast_5 - Ip_arr > T) + \
              (fast_9 - Ip_arr > T) + \
              (fast_13 - Ip_arr > T)

    #print(less, greater)
    X, Y = np.mgrid[-R:R + 1, -R:R + 1]
    mask = (X ** 2 + Y ** 2 <= R ** 2 + 1) * ((R - 1) ** 2 + 1 < X ** 2 + Y ** 2)


    #show_image(mask)
    #print(mask.shape)
    #print(grey_image.shape)
    #print(np.squeeze(np.where((less > 2).reshape(-1))))
    for i in np.squeeze(np.where((less > 2).reshape(-1))):
        x, y = i // (W - 2 * R), i % (W - 2 * R)
        '''
        проверка противоположных не работает:
           1 1 1
         0   |   1
        1    |    0
        1---------0
        1    |    0
         1   |   1
           1 1 1
        '''
        '''
        print(x, y, (grey_image[x: x + 2 * R + 1, y: y + 2 * R + 1])[mask].shape)
        temp = (Ip_arr[x, y] - grey_image[x: x + 2 * R + 1, y: y + 2 * R + 1][mask] > T).astype(np.int8)
        s = temp.sum()
        if s > 14:
            res.append((x+R, y+R))
            continue
        if temp.sum() < 12:
            continue
        '''
        count = 0
        # print(grey_image[x: x + 2 * R + 1, y: y + 2 * R + 1])
        first_seq = None
        last_check = True
        temp = Ip_arr[x, y] - grey_image[x: x + 2 * R + 1, y: y + 2 * R + 1] * mask > T
        for ix, iy in pixels:
            if (temp[ix + R, iy + R] == 1):
                count += 1
            else:
                if first_seq is None:
                    first_seq = count
                count = 0

            if (count == 12):
                res.append((x + R, y + R))
                last_check = False
                break
        if last_check and count + first_seq >= 12:
            res.append((x + R, y + R))


    for i in np.squeeze(np.where((greater > 2).reshape(-1))):
        x, y = i // (W - 2 * R), i % (W - 2 * R)
        '''
        print(grey_image[x: x + 2 * R + 1, y: y + 2 * R + 1][mask])
        temp = (grey_image[x: x + 2 * R + 1, y: y + 2 * R + 1][mask] - Ip_arr[x, y] > T).astype(np.int8)
        s = temp.sum()
        if s > 14:
            res.append((x+R, y+R))
            continue
        if temp.sum() < 12:
            continue
            '''
        count = 0
        #print(grey_image[x: x + 2 * R + 1, y: y + 2 * R + 1])
        first_seq = None
        last_check = True
        temp = grey_image[x: x + 2 * R + 1, y: y + 2 * R + 1] * mask - Ip_arr[x, y] > T
        for ix, iy in pixels:
            if (temp[ix + R, iy + R] == 1):
                count += 1
            else:
                if first_seq is None:
                    first_seq = count
                count = 0

            if (count == 12):
                res.append((x + R, y + R))
                last_check = False
                break
        if last_check and count + first_seq >= 12:
            res.append((x + R, y + R))

    return res






if __name__ == "__main__":
    img = cv2.imread(IMAGE_PATH)
    #img = np.zeros((10, 10, 3), dtype=np.uint8) + 128
    #img[3:7,3:7,:] = np.array([0,0,0])
    angles = FAST(img)
    color = (0, 0, 255)
    for y, x in angles:
        color = (0, 0, 255)
        cv2.circle(img, (x, y), 10, color, 2)
    show_image(img)

