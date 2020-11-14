from config import *
from matplotlib import pyplot as plt
import sklearn.linear_model as lm

OBJ_PATH = r'box.png'
SCENE_PATH = r'box_in_scene.png'
DRAW_VECTORS = True
BBF_THRESHOLD = 0.7
USE_SCALE_FOR_CENTER_VECTORS = True
DO_NOT_USE_SCALE_FOR_ACCUMULATOR = True
USE_BFF = True
COORD_STEP = 20
ORB_SCALE_FACTOR = 1.2
ORB_NLEVELS = 8
CENTER_VECS_COLOR = (255, 0, 0)
MATCH_COLOR = (255, 255, 0)

def drawMatches(obj, kp1, scene, kp2, matches, mask=None):
    if obj.ndim == 2:
        obj = cv2.cvtColor(obj, cv2.COLOR_GRAY2BGR)
    if scene.ndim == 2:
        scene = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    oH, oW, _ = obj.shape
    sH, sW, _ = scene.shape
    img = np.zeros((max(oH, sH), oW+sW, 3), dtype=np.uint8)
    img[:oH, :oW, :] = obj
    img[:sH, oW:, :] = scene
    for match in matches:
        obj_kp = kp1[match.queryIdx]
        scene_kp = kp2[match.trainIdx]
        cv2.line(img, (int(obj_kp.pt[0]), int(obj_kp.pt[1])),
                        (int(scene_kp.pt[0] + oW), int(scene_kp.pt[1])), MATCH_COLOR, 1)
    return img


def hamming_distance(vec1, vec2):
    assert vec1.ndim == 1
    return (vec1 ^ vec2).sum(axis=1)


class match():
    def __init__(self, queryIdx, trainIdx, distance=0):
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.distance = distance


ADD_CONST = 100
def bbf(desc1, desc2):
    desc1 = np.unpackbits(desc1,axis=1)
    desc2 = np.unpackbits(desc2, axis=1)
    distances = np.array([hamming_distance(vec, desc2) for vec in desc1])
    fm_i = np.argmin(distances, axis=1)
    fm_v = distances[np.arange(desc1.shape[0]), fm_i]
    distances[np.arange(desc1.shape[0]), fm_i] += ADD_CONST
    sm_v = np.min(distances, axis=1)
    indices = np.where(fm_v / sm_v < BBF_THRESHOLD)[0]
    out = []
    for ind in indices:
        out.append(match(ind, fm_i[ind], distance=fm_v[ind]))
    return out



class Localizator():
    def __init__(self):
        self.scale_factor = ORB_SCALE_FACTOR
        self.nlevels = ORB_NLEVELS
        self.angle_step = 12
        self.scene_y_step = COORD_STEP
        self.scene_x_step = COORD_STEP
        self.orb = cv2.ORB_create(scaleFactor=self.scale_factor, nlevels=self.nlevels)

    def __call__(self, object, scene):
        scene_H, scene_W, _ = scene.shape
        obj_H, obj_W, _ = object.shape
        obj_center = (obj_W / 2, obj_H / 2)
        kp1, des1 = self.orb.detectAndCompute(object, None)
        kp2, des2 = self.orb.detectAndCompute(scene, None)
        if USE_BFF:
            matches = bbf(des1, des2) #bf.match(des1, des2)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)


        #matches = sorted(matches, key=lambda x: x.distance) #[:self.match_count]
        print(f"Matched {len(matches)} keypoints")
        img = drawMatches(object, kp1, scene, kp2, matches, None)
        plt.imshow(img), plt.show()

        self.nlevels = 0
        accum = [[[[[] for _ in range(0, scene_W, self.scene_x_step)]
                   for _ in range(0, scene_H, self.scene_y_step)]
                  for _ in range(0, 361, self.angle_step)]
                 for _ in range(-self.nlevels, self.nlevels + 1)]

        max_len = 0
        max_features = 0
        max_keypoints = None
        more_than3 = set()
        for match in matches:
            obj_kp = kp1[match.queryIdx]
            scene_kp = kp2[match.trainIdx]
            delta_theta = scene_kp.angle - obj_kp.angle
            if USE_SCALE_FOR_CENTER_VECTORS:
                scale = scene_kp.octave - obj_kp.octave
            else:
                scale = 0
            center_vec = obj_center[0] - obj_kp.pt[0], obj_center[1] - obj_kp.pt[1]
            if DRAW_VECTORS:
                cv2.line(object, (int(obj_kp.pt[0]), int(obj_kp.pt[1])),
                     (int(obj_kp.pt[0] + center_vec[0]), int(obj_kp.pt[1] + center_vec[1])), CENTER_VECS_COLOR, 1)

            center_vec = (
                np.array(center_vec).dot(
                    cv2.getRotationMatrix2D(center=(0, 0), angle=delta_theta,
                                            scale=(self.scale_factor ** scale))[:, :2]))
            if DRAW_VECTORS:
                cv2.line(scene, (int(scene_kp.pt[0]), int(scene_kp.pt[1])),
                        (int(scene_kp.pt[0] + center_vec[0]), int(scene_kp.pt[1] + center_vec[1])), CENTER_VECS_COLOR, 1)


            if DO_NOT_USE_SCALE_FOR_ACCUMULATOR:
                scale =0

            features = (scale, int(delta_theta // self.angle_step), int((scene_kp.pt[1] + center_vec[1]) // self.scene_y_step),
                        int((scene_kp.pt[0] + center_vec[0]) // self.scene_x_step))

            try:
                curr_list = accum[features[0]][features[1]][features[2]][features[3]]
                # noinspection PyTypeChecker
                curr_list.append((obj_kp, scene_kp))
                if len(curr_list) > max_len:
                    max_keypoints = curr_list
                    max_features = features
                    max_len = len(curr_list)
                if len(curr_list) > 2:
                    more_than3.add(features)
            except IndexError as e:
                continue # bad match?? point + center_vec

        print(f"MAX FUTURES: (scale, theta, y, x): {max_features}, points_count: {max_len}")
        if max_len < 3:
            raise Exception("NEED MORE POINTS")
        #scene = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
        #cv2.drawContours(scene, [rect_new], 0, (0, 255, 0), 2)

        #img = cv2.drawMatches(object, kp1, scene, kp2, matches[:4], None)
        #plt.imshow(img), plt.show()

        kp_dict = {f: accum[f[0]][f[1]][f[2]][f[3]] for f in more_than3}
        skm = lm.LinearRegression(fit_intercept=False)
        max_size = 4
        x = None
        while True:
            kps = sum([kp_dict[f] for f in kp_dict if len(kp_dict[f]) > 2], [])
            if len(kps) == 0:
                break
            A, b = [], []
            for obj_kp, scene_kp in kps:
                A.append([obj_kp.pt[0], obj_kp.pt[1], 0, 0, 1, 0])
                A.append([0, 0, obj_kp.pt[0], obj_kp.pt[1], 0, 1])
                b.append(scene_kp.pt)
            A = np.array(A)
            b = np.array(b).reshape(-1, 1)
            skm.fit(A, b)
            x = skm.coef_.reshape((-1, 1))
            indices = np.where(np.any((A.dot(x) - b > max_size / 4).reshape(-1, 2), axis=1))[0]
            if indices.size == 0:
                break
            delete_candidates = [kps[i] for i in indices]
            for key in kp_dict:  # also walk on list with size=2
                for kp in kp_dict[key]:
                    if kp in delete_candidates:
                        kp_dict[key].remove(kp)


        rect_kp = np.array([(0, 0), (obj_W, 0), (obj_W, obj_H),
                            (0, obj_H)])

        rect_new = np.int0(rect_kp.dot((x[:4].reshape((2, 2)).T)) + x[4:].reshape((1, 2)))

        cv2.drawContours(scene, [rect_new], 0, (0, 255, 0), 2)
        img = drawMatches(object, kp1, scene, kp2, matches[:0], None)
        plt.imshow(img), plt.show()










if __name__ == "__main__":
    grey_obj = cv2.imread(OBJ_PATH)
    #grey_obj = cv2.cvtColor(grey_obj, cv2.COLOR_BGR2GRAY)
    grey_scene = cv2.imread(SCENE_PATH)
    #grey_scene = cv2.cvtColor(grey_scene, cv2.COLOR_BGR2GRAY)

    loc = Localizator()
    loc(grey_obj, grey_scene)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(grey_obj, None)
