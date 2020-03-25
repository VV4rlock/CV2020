from config import *

from Harris_corner_detector import get_harris_response
from FAST_corner_detector import oriented_FAST
from BRIEF import RotatedBRIEF

class ORB:
    def __init__(self, S=31, bitsize=256, fast_radius=9, oriented_FAST_radius=20, FAST_threshold = 20, Harris_threshold=110):
        self.FAST_Radius = fast_radius
        self.OFAST_Radius = oriented_FAST_radius
        self.S = S
        self.bitsize = bitsize
        self.FAST_threshold = FAST_threshold
        self.nms_zone = self.FAST_Radius
        self.get_descriptors = RotatedBRIEF(S=S, bitsize=bitsize, mode="GIII")
        self.Harris_threshold = Harris_threshold
        self.offset = max(int(self.S * 2**.5 // 2) + 3, self.FAST_Radius, self.OFAST_Radius)
        #self.scales = [1, 1/2**0.5, 1/2, 1/(2*2**0.5)]
        self.scales = [1, 1 / 2 ** 0.5, 1 / 2, 1 / (2 * 2 ** 0.5), 1/4 , 1/(4*2**0.5), 1/8]

    def NMS(self, image: np.ndarray, n=9) -> np.ndarray:
        H, W = image.shape
        for k in np.where((image != 0).reshape(-1))[0]:
            i, j = k // W, k % W
            if image[i, j] != image[max(0, i - n): min(i + n, W), max(0, j - n): min(j + n, W)].max():
                image[i, j] = 0

        return image

    def separate_by_harris_response(self, harris_response, angles_coords, theta, harris_threshhold=30):
        assert len(angles_coords) == len(theta)

        temp_harris = np.zeros_like(harris_response)
        temp_theta = np.zeros_like(harris_response, dtype=np.float32)
        coord_tuple = (angles_coords[:, 0], angles_coords[:, 1])
        temp_harris[coord_tuple] = harris_response[coord_tuple]
        temp_theta[coord_tuple] = theta
        temp_harris[temp_harris < harris_threshhold] = 0

        temp_harris = self.NMS(temp_harris, n=self.nms_zone)
        w = temp_harris.shape[1]
        # responses is needed ?
        nonzero_indexes = np.where((temp_harris != 0).reshape(-1))[0]
        angles_out = np.stack((nonzero_indexes // w, nonzero_indexes % w), axis=1)
        theta_out = temp_theta[temp_harris != 0]

        return angles_out, theta_out

    def __call__(self, image):
        res = np.zeros((0, 2), dtype=np.uint32)
        descriptors_out = np.zeros((0, self.bitsize), dtype=np.uint8)
        for scale in self.scales:
            x_resize, y_resize = int(image.shape[1] * scale), int(image.shape[0] * scale)
            if x_resize < self.offset * 2 or y_resize < self.offset * 2:
                continue
            img = cv2.resize(image, (x_resize, y_resize))

            img_for_show = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            harris = get_harris_response(img)
            angles, theta = oriented_FAST(img, R=self.OFAST_Radius, fast_radius=self.FAST_Radius,
                                          fast_threshold=self.FAST_threshold, offset=self.offset)
            for coord in angles:
                img_for_show[coord[0], coord[1]] = (255, 0, 0)
            if len(angles) == 0:
                continue
            angles, theta = self.separate_by_harris_response(harris, angles, theta, harris_threshhold=self.Harris_threshold)
            for coord in angles:
                img_for_show[coord[0], coord[1]] = (0, 0, 255)
            if len(angles) == 0:
                continue
            descriptors = self.get_descriptors(img, angles, theta)

            print(f"scale {scale}")
            for coord in angles:
                #print(coord)
                cv2.circle(img_for_show, (coord[1], coord[0]), 10, (0, 255, 0), 1)
            show_image(img_for_show)
            angles = (angles * (image.shape[0] / y_resize, image.shape[1] / x_resize)).astype(np.uint32)
            #angles = (angles / scale).astype(np.uint32)
            res = np.concatenate((res, angles), axis=0)
            #print(f"restored: \n{res}")
            descriptors_out = np.concatenate((descriptors_out, descriptors), axis=0)

        img_for_show =  image.copy()
        img_for_show = cv2.cvtColor(img_for_show, cv2.COLOR_GRAY2BGR)
        for coord in res:
            cv2.circle(img_for_show, (coord[1], coord[0]), 10, (0, 255, 0), 1)
        show_image(img_for_show)
        return res, descriptors_out


if __name__ == "__main__":
    img_input = cv2.imread(IMAGE_PATH)
    grey_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    orb = ORB()
    grey_image = np.pad(grey_image, 20, mode='constant')

    image_center = tuple(np.array(grey_image.shape[1::-1]) / 2)
    grey_image_rotated = cv2.warpAffine(grey_image, cv2.getRotationMatrix2D(image_center, 30, 1.0) , grey_image.shape[1::-1], flags=cv2.INTER_LINEAR)
    mul=0.8
    grey_image_rotated = cv2.resize(grey_image_rotated, (int(grey_image_rotated.shape[1]*mul), int(grey_image_rotated.shape[0]*mul)))
    print(grey_image_rotated.shape)
    pad_add = ((grey_image.shape[0] - grey_image_rotated.shape[0])//2,
                                                    (grey_image.shape[1] - grey_image_rotated.shape[1])//2 )
    grey_image_rotated = np.pad(grey_image_rotated, ((pad_add[0],grey_image.shape[0] - grey_image_rotated.shape[0] - pad_add[0]),
                                                     (pad_add[1],(grey_image.shape[1] - grey_image_rotated.shape[1] - pad_add[1]))),
                                                     mode='constant')

    angles1, descriptors1 = orb(grey_image)
    angles2, descriptors2 = orb(grey_image_rotated)

    res_image = np.concatenate((grey_image, grey_image_rotated), axis=1)

    '''
    res_image = cv2.cvtColor(res_image, cv2.COLOR_GRAY2BGR)
    offset = grey_image.shape[1]
    for coord in angles1:
        cv2.circle(res_image, (coord[1], coord[0]), 10, (0, 255, 0), 1)
    for coord in angles2:
        cv2.circle(res_image, (coord[1] + offset, coord[0]), 10, (0, 255, 0), 1)
    '''
    #sift = cv2.ORB_create()
    # find the keypoints and descriptors with SIFT
    #kp1, des1 = sift.detectAndCompute(grey_image, None)
    #print(des1)
    kp1 = [cv2.KeyPoint(coords[1], coords[0], 1) for coords in angles1]
    kp2 = [cv2.KeyPoint(coords[1], coords[0], 1) for coords in angles2]
    descriptors1 = np.array(descriptors1)
    descriptors2 = np.array(descriptors2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(grey_image, kp1, grey_image_rotated, kp2, matches[:10], res_image, flags=2)

    show_image(img3)