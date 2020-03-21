from config import *
from FAST_corner_detector import oriented_FAST

class RotatedBRIEF:
    modes = ["GI", "GIII"]

    def __init__(self, S, bitsize, mode="GIII"):
        np.random.seed(1)
        assert mode in self.modes
        self.R = S // 2 - 1
        X = np.zeros((bitsize, 2))
        Y = np.zeros((bitsize, 2))
        if mode == self.modes[0]:
            x_random_func = lambda: np.random.uniform(-S / 2, S / 2, (1, 2))
            y_random_func = lambda: np.random.uniform(-S / 2, S / 2, (1, 2))
        elif mode == self.modes[1]:
            x_random_func = lambda: np.random.normal(0, S * S / 25, (1, 2))
            y_random_func = lambda: np.random.normal(0, S * S / 100, (1, 2))
        else:
            raise Exception("WTF?!")
        index = 0
        while index < bitsize:
            x = x_random_func()
            y = y_random_func()
            if max(np.abs(x).max(), np.abs(x).max()) < self.R:
                X[index] = x
                Y[index] = y
                index += 1

        rot_matrix_by_angle = {a: -cv2.getRotationMatrix2D(center=(0, 0), angle=a, scale=1)[:, :2]
                               for a in range(0, 361, 12)}
        self.coord_by_andle = [(
                                np.round(np.dot(X, rot_matrix_by_angle[key]).astype(np.int8)),
                                np.round(np.dot(Y, rot_matrix_by_angle[key]).astype(np.int8))
                                         ) for key in rot_matrix_by_angle
                                ]
        rad = np.pi / 180
        self.angles_array = np.array([a * rad for a in rot_matrix_by_angle])

    def __call__(self, image, angles_coord, theta):
        assert len(angles_coord) == len(theta)
        desctiptors = []
        for i, coord in enumerate(angles_coord):
            coord = angles_coord[i]
            patch1, patch2 = self.coord_by_andle[np.abs(self.angles_array - theta[i]).argmin()]
            keypoint_pos1 = (patch1[:, 1] + coord[0], patch1[:, 0] + coord[1])
            keypoint_pos2 = (patch2[:, 1] + coord[0], patch2[:, 0] + coord[1])
            #print(f"coord: {coord}\n{keypoint_pos1}\n{np.abs(self.angles_array - theta[i]).argmin()}\n{patch1}")
            descriptor = image[keypoint_pos1] < image[keypoint_pos2]
            desctiptors.append(descriptor)
        return np.array(desctiptors, dtype=np.uint8)


if __name__ == "__main__":
    img_input = cv2.imread(IMAGE_PATH)
    grey_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY).astype(np.int16)
    angles, theta = oriented_FAST(grey_image)
    BRIEF = RotatedBRIEF(9, 32)
    descriptors = BRIEF(grey_image, angles, theta)
    print(descriptors)
