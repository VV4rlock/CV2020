from config import *
from FAST_corner_detector import oriented_FAST

class RotatedBRIEF:
    modes = ["GI", "GIII"]

    def __init__(self, S, bitsize, mode="GI"):
        np.random.seed(1)
        assert mode in self.modes
        self.R = S // 2
        if mode == self.modes[0]:
            X = np.random.uniform(-S / 2, S / 2, (bitsize, 2))
            Y = np.random.uniform(-S / 2, S / 2, (bitsize, 2))
        elif mode == self.modes[1]:
            X = np.random.normal(0, S * S / 25, (bitsize, 2))
            Y = np.random.normal(0, S * S / 100, (bitsize, 2))
        else:
            raise Exception("WTF?!")
        X, Y = X.astype(np.int8), Y.astype(np.int8)
        new_pos1 = np.stack([X[:, 1], X[:, 0], np.zeros(len(X))], axis=1)
        new_pos2 = np.stack([Y[:, 1], Y[:, 0], np.zeros(len(Y))], axis=1)
        rot_matrix_by_angle = {a: cv2.getRotationMatrix2D(center=(0, 0), angle=a, scale=1) for a in range(0, 361, 12)}
        rad = np.pi / 180
        self.coord_by_andle = [(
                                np.round(np.dot(new_pos1, rot_matrix_by_angle[key].T).astype(np.int8)),
                                np.round(np.dot(new_pos2, rot_matrix_by_angle[key].T).astype(np.int8))
                                         )
                          for key in rot_matrix_by_angle
                        ]
        self.angles_array = np.array([a* rad for a in rot_matrix_by_angle])

    def __call__(self, image, angles_coord, theta):
        assert len(angles_coord) == len(theta)
        desctiptors = []
        for i in range(len(angles_coord)):
            coord = angles_coord[i]
            patch1, patch2 = self.coord_by_andle[np.abs(self.angles_array - theta[i]).argmin()]
            keypoint_pos1 = (patch1[:, 1] + coord[0], patch1[:, 0] + coord[1])
            keypoint_pos2 = (patch2[:, 1] + coord[0], patch2[:, 0] + coord[1])
            descriptor = image[keypoint_pos1] < image[keypoint_pos2]
            desctiptors.append(descriptor.astype(np.uint8))
        return desctiptors


if __name__ == "__main__":
    img_input = cv2.imread(IMAGE_PATH)
    grey_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY).astype(np.int16)
    less, greater = oriented_FAST(grey_image)
    BRIEF = RotatedBRIEF(9, 32)
    less_descriptors = BRIEF(grey_image, less)
    print(less_descriptors)
