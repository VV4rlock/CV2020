from config import *

from Harris_corner_detector import harris_response
from FAST_corner_detector import oriented_FAST
from BRIEF import RotatedBRIEF

class ORB:
    def __init__(self, S=31, bitsize=32, fast_radius=9, oriented_FAST_radius=15):
        self.FAST_Radius = fast_radius
        self.OFAST_Radius = oriented_FAST_radius
        self.FAST_threshold = 100
        self.get_descriptors = RotatedBRIEF(S, bitsize)
        self.Harris_threshold = 100
        self.scales = [1, 1/2, 1/4, 1/8]

    def similar(self, desc1, desc2):
        HW = (desc1 ^ desc2).sum()
        ones_similar = (desc1 & desc2).sum()
        zeros_similar = ((desc1 ^ 1) & (desc2 ^ 1)).sum()


    @staticmethod
    def NMS(image: np.ndarray, n=20) -> np.ndarray:
        H, W = image.shape
        try:
            for k in np.squeeze(np.where((image != 0).reshape(-1))):
                i, j = k // W, k % W
                if image[i, j] != image[max(0, i - n): min(i + n, W), max(0, j - n): min(j + n, W)].max():
                    image[i, j] = 0
        finally:
            return image


    @staticmethod
    def get_response(harris_response, angles_coords, theta, harris_threshhold=30):
        temp = np.zeros_like(harris_response)
        for coords in angles_coords:
            temp[coords[0], coords[1]] = harris_response[coords[0], coords[1]]
        temp[temp < harris_threshhold] = 0
        temp = ORB.NMS(temp)
        angles, responses, theta_out = [], [], []
        for index, coords in enumerate(angles_coords):
            if temp[coords[0], coords[1]] != 0:
                angles.append(coords)
                responses.append(temp[coords[0], coords[1]])
                theta_out.append(theta[index])
        return angles, theta_out, responses


    def __call__(self, image):
        scales = [1, 1 / 2, 1 / 4, 1 / 8]
        res = np.zeros((0, 2), dtype=np.uint8)
        descriptors_out = []
        for scale in scales:
            img = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

            harris = harris_response(img)
            less, less_theta, greater, greater_theta = oriented_FAST(img, R=self.OFAST_Radius,
                                                                 fast_radius=self.FAST_Radius, fast_threshold=50, offset=15)
            angles = less+greater
            if not angles:
                continue
            theta = np.append(less_theta, greater_theta)
            angles, theta, responses = ORB.get_response(harris, angles, theta, harris_threshhold=100)
            if not angles:
                continue
            descriptors_out += self.get_descriptors(grey_image, angles, theta)

            res = np.concatenate((res, (np.array(angles) / scale).astype(np.uint8)), axis=0)
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            #print(res)
            for coord in res:
                #print(coord)
                cv2.circle(img, (coord[1], coord[0]), 10, (0, 255, 0), 1)
            show_image(img)
            #print(res)


        return res, descriptors_out


if __name__ == "__main__":
    img_input = cv2.imread(IMAGE_PATH)
    grey_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    orb = ORB()
    angles1, descriptors1 = orb(grey_image)

    image_center = tuple(np.array(grey_image.shape[1::-1]) / 2)
    grey_image_rotated = cv2.warpAffine(grey_image, cv2.getRotationMatrix2D(image_center, 45, 1.0) , grey_image.shape[1::-1], flags=cv2.INTER_LINEAR)
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
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(grey_image, kp1, grey_image_rotated, kp2, matches[:10], res_image,flags=2)

    show_image(img3)