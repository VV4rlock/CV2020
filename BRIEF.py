from config import *

class RotatedBRIEF:
    modes = ["GI", "GIII"]

    def __init__(self, S=31, bitsize=256, mode="GIII", file_with_patch=None):
        np.random.seed(1)
        assert mode in self.modes
        self.R = S // 2
        self.S = S
        self.X = np.zeros((bitsize, 2))
        self.Y = np.zeros((bitsize, 2))
        if file_with_patch is None:
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
        else:
            with open(file_with_patch, "rb") as f:
                import pickle
                self.X, self.Y = pickle.load(f)

        if PRINT_PATCH:
            mul = 27
            patch_image = np.zeros((S*mul, S*mul))

            for x,y in np.stack(((self.X.astype(np.int32) + self.R) * mul, (self.Y.astype(np.int32) + self.R) * mul), axis=1):
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

    def hamming_correlation(self, test_set, candidate, n):
        return np.abs(np.sum(test_set ^ candidate.reshape(-1, 1), axis=0) / n - 0.5).max()

    first = None
    def get_decorrelated_tests(self, test_patches, bitsize=128):
        print(f"Decorrelated test procedure started")
        threshold = .17
        test_count = 10000
        T_x, T_y = np.zeros((test_count, 2)), np.zeros((test_count, 2))
        index = 0
        print(f"generating tests")
        while index < test_count:
            x, y = self.random_func(1)
            if max(np.abs(x).max(), np.abs(y).max()) <= self.R:
                T_x[index] = x
                T_y[index] = y
                index += 1
        T_x, T_y = np.round(T_x).astype(np.int8), np.round(T_y).astype(np.int8)
        test_patches_len = len(test_patches)
        test_set = np.zeros((test_patches_len, test_count), dtype=np.int8)
        for i, test_patch in enumerate(test_patches):
            test_set[i] = test_patch[(T_x[:, 0] + self.R, T_x[:, 1] + self.R)] < test_patch[
                (T_y[:, 0] + self.R, T_y[:, 1] + self.R)]
        test_sum = np.sum(test_set, axis=0).astype(np.float64)
        test_sum /= test_patches_len
        indexes = np.argsort(np.abs(test_sum - .5))
        while 1:
            R = [indexes[0]]
            l = 1
            for i, index in enumerate(indexes[1:]):
                cor = self.hamming_correlation(test_set[:, R], test_set[:, index], test_patches_len)
                #print(f"cor {index} = {cor}")
                if cor < threshold:
                    R += [index]
                    l += 1
                    if l == bitsize:
                        print()
                        if True:
                            import matplotlib.pyplot as plt
                            import matplotlib as mpl
                            import matplotlib.colors as colors
                            import matplotlib.cm as cm
                            import matplotlib.pylab as pl

                            cNorm = colors.Normalize(vmin=0, vmax=0.5)  # re-wrapping normalization
                            scalarMap = cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('plasma'))

                            #print(f"wtf: {scalarMap.autoscale()}")
                            scalarMap.set_array(np.round(np.linspace(0.01, 0.5, 50),2))
                            #norm = mpl.colors.Normalize(vmin=0, vmax=0.5)
                            #colormap = {j: (.5 + j , 0., .1 - 2 * j ) for j in np.round(np.linspace(0.01, 0.5, 50),2)}
                            #print(colormap)
                            plt.subplot(1, 2, 1)
                            plt.title(f"first tests ({bitsize})")
                            index_set = set(range(bitsize))
                            for j in index_set:
                                correlation = np.round(
                                             self.hamming_correlation(test_set[:, list(index_set - {j})], test_set[:, j],
                                                                      test_patches_len), 2)
                                plt.plot((T_x[j][1] + self.R, T_y[j][1] + self.R), (T_x[j][0] + self.R, T_y[j][0] + self.R),
                                         color=scalarMap.to_rgba(correlation))

                            plt.subplot(1, 2, 2)
                            index_set = set(R)
                            for j in index_set:
                                correlation = np.round(
                                    self.hamming_correlation(test_set[:, list(index_set - {j})], test_set[:, j],
                                                             test_patches_len), 2)
                                plt.plot((T_x[j][1] + self.R, T_y[j][1] + self.R),
                                         (T_x[j][0] + self.R, T_y[j][0] + self.R),
                                         color=scalarMap.to_rgba(correlation))

                            plt.colorbar(mappable=scalarMap)
                            plt.title(f"decorelated ({bitsize})")
                            plt.show()
                                #cv2.line(patch_image, (x[1], x[0]), (y[1], y[0]), 100, 1)

                        return T_x[R], T_y[R]
                print(f"\rChecked {i+2}/{test_count} tests. {l}/{bitsize} tests found. current_normolised_weidth={test_sum[index]}", end="")
            threshold *= (1 + (bitsize-l)/bitsize)
            print(f"\nNOT FOUND ENOUGHT TESTS: trying with threshold={threshold}")


    def construct_decorelted_patch(self):
        from ORB import ORB
        import pickle
        if GENERATE_TEST_PATHCES:
            orb = ORB()
            test_patches = orb.get_test_patches()
            with open(DUMP_TESTPATCHES_FILENAME, "wb") as f:
                pickle.dump(test_patches, f)
            print(f"test_patches was dumped to {DUMP_TESTPATCHES_FILENAME}")
        else:
            with open(DUMP_TESTPATCHES_FILENAME, 'rb') as f:
                test_patches = pickle.load(f)
        T_x, T_y = self.get_decorrelated_tests(test_patches, bitsize=BRIEF_BITSIZE)
        with open(f"patch{BRIEF_BITSIZE}.pickle", "wb") as f:
            pickle.dump((T_x, T_y), f)



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
    br = RotatedBRIEF(mode="GI")
    br.construct_decorelted_patch()
    #from ORB import ORB
    #orb = ORB()
    #orb.get_test_patches()
