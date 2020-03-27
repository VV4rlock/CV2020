from config import *
from FAST_corner_detector import oriented_FAST

PRINT_PATCH = False
class RotatedBRIEF:
    modes = ["GI", "GIII"]

    def __init__(self, S=31, bitsize=256, mode="GIII"):
        np.random.seed(1)
        assert mode in self.modes
        self.R = S // 2
        self.S = S
        self.X = np.zeros((bitsize, 2))
        self.Y = np.zeros((bitsize, 2))
        if mode == self.modes[0]:
            self.random_func = lambda x: (np.random.uniform(-S / 2, S / 2, (x,2)), np.random.uniform(-S / 2, S / 2, (x,2)))
        elif mode == self.modes[1]:
            def random_func(x):
                res0 = np.random.normal(0, S * S / 25, (x,2))
                return res0, np.random.normal(res0, S*S/100, (x,2))
            self.random_func=random_func
        else:
            raise Exception("WTF?!")
        index = 0
        while index < bitsize:
            x, y = self.random_func(1)
            if max(np.abs(x).max(), np.abs(y).max()) <= self.R:
                self.X[index] = x
                self.Y[index] = y
                index += 1

        if PRINT_PATCH:
            mul = 27
            patch_image = np.zeros((S*mul, S*mul))

            for x,y in np.stack(((((self.X + self.R) * mul)).astype(np.uint32), ((self.Y + self.R) * mul).astype(np.uint32)), axis=1):
                cv2.line(patch_image, (x[1],x[0]), (y[1],y[0]), 100, 1)
                patch_image[(x[0], x[1])] = 255
                patch_image[(y[0], y[1])] = 255
            show_image(patch_image)



        rot_matrix_by_angle = {a: cv2.getRotationMatrix2D(center=(0, 0), angle=a, scale=1)[:, :2].T
                               for a in range(0, 361, 12)}
        self.coord_by_andle = [(
                                np.round(np.dot(self.X, rot_matrix_by_angle[key]).astype(np.int8)),
                                np.round(np.dot(self.Y, rot_matrix_by_angle[key]).astype(np.int8))
                                         ) for key in rot_matrix_by_angle
                                ]
        rad = np.pi / 180
        self.angles_array = np.array([a * rad for a in rot_matrix_by_angle])

    def is_decorelated(self, decorelated_tests, test):
        for test in ['place_for_tests_patches']:
            pass

    def hamming_correlation(self, x, n):
        return max([max(np.abs(np.sum(x ^ raw[:i], axis=1) / n - .5).max(), np.abs(np.sum(x ^ raw[i + 1:], axis=1) / n - .5).max())
                                            for i, raw in enumerate(x)])

    def get_decorelated_tests(self, test_patches, bitsize=128):
        gamma = 0.1
        test_count = 1000
        T_x, T_y = np.random.uniform(-self.S / 2, self.S / 2, (test_count,2)), np.random.uniform(-self.S / 2, self.S / 2, (test_count,2))
        T_x, T_y = np.round(T_x).astype(np.int8), np.round(T_y).astype(np.int8)
        test_patches_len = len(test_patches)
        test_set = np.zeros((test_patches_len, test_count), dtype=np.float32)
        for i, test_patch in enumerate(test_patches):
            test_set[i] = test_patch[T_x] < test_patch[T_y]
        test_sum = np.sum(test_set, axis=0)
        test_sum /= test_patches_len
        indexes = np.argsort(np.abs(test_sum - .5))
        R = [indexes[0]]
        l = 1
        for index in indexes[1:]:
            if self.hamming_correlation(test_set[:, R + [index]], l + 1) < gamma:
                R += [index]
                l += 1
                if l == bitsize:
                    pass






    @staticmethod
    def get_integral_image(image):
        return np.cumsum(np.cumsum(image.astype(np.float64), axis=1), axis=0)

    def __call__(self, image, angles_coord, theta):
        assert len(angles_coord) == len(theta)
        desctiptors = []
        integral = self.get_integral_image(image) / 25
        for i, coord in enumerate(angles_coord):
            coord = angles_coord[i]
            patch1, patch2 = self.coord_by_andle[np.abs(self.angles_array - theta[i]).argmin()]
            #keypoint_pos1 = (patch1[:, 0] + coord[0], patch1[:, 1] + coord[1])
            #keypoint_pos2 = (patch2[:, 0] + coord[0], patch2[:, 1] + coord[1])
            #print(f"coord: {coord}\n{keypoint_pos1}\n{np.abs(self.angles_array - theta[i]).argmin()}\n{patch1}")
            mid1 = integral[(patch1[:, 0] + coord[0] + 2, patch1[:, 1] + coord[1] + 2)] - \
                integral[(patch1[:, 0] + coord[0] + 2, patch1[:, 1] + coord[1] - 3)] - \
                integral[(patch1[:, 0] + coord[0] - 3, patch1[:, 1] + coord[1] + 2)] + \
                integral[(patch1[:, 0] + coord[0] - 3, patch1[:, 1] + coord[1] - 3)]
            mid2 = integral[(patch2[:, 0] + coord[0] + 2, patch2[:, 1] + coord[1] + 2)] - \
                   integral[(patch2[:, 0] + coord[0] + 2, patch2[:, 1] + coord[1] - 3)] - \
                   integral[(patch2[:, 0] + coord[0] - 3, patch2[:, 1] + coord[1] + 2)] + \
                   integral[(patch2[:, 0] + coord[0] - 3, patch2[:, 1] + coord[1] - 3)]
            descriptor = mid1 < mid2
            desctiptors.append(descriptor)
        return np.array(desctiptors, dtype=np.uint8)


if __name__ == "__main__":
    img_input = cv2.imread(IMAGE_PATH)
    grey_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY).astype(np.int16)
    angles, theta = oriented_FAST(grey_image)
    BRIEF = RotatedBRIEF(9, 32)
    descriptors = BRIEF(grey_image, angles, theta)
    print(descriptors)
