from config import *
from matplotlib import pyplot as plt
import sklearn.linear_model as lm

OBJ_PATH = r'box.png'
SCENE_PATH = r'box_in_scene.png'
DRAW_VECTORS = False

class Localizator():
    def __init__(self):
        self.scale_factor = 1.2
        self.nlevels = 8
        self.angle_step = 12
        self.scene_y_step = 10
        self.scene_x_step = 10
        self.match_count = 5
        self.orb = cv2.ORB_create(scaleFactor=self.scale_factor, nlevels=self.nlevels)

    def __call__(self, object, scene):
        scene_H, scene_W = scene.shape
        obj_H, obj_W = object.shape
        obj_center = (obj_W / 2, obj_H / 2)
        kp1, des1 = self.orb.detectAndCompute(object, None)
        kp2, des2 = self.orb.detectAndCompute(scene, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance) #[:self.match_count]

        accum = [[[[[] for _ in range(0, scene_W, self.scene_x_step)]
                   for _ in range(0, scene_H, self.scene_y_step)]
                  for _ in range(0, 361, self.angle_step)]
                 for _ in range(-self.nlevels, self.nlevels + 1)]

        max_len = 0
        max_features = 0
        max_keypoints = None
        more_than3 = set()
        print(scene_H, scene_W)
        for match in matches:
            obj_kp = kp1[match.queryIdx]
            scene_kp = kp2[match.trainIdx]
            delta_theta = scene_kp.angle - obj_kp.angle
            scale = scene_kp.octave - obj_kp.octave
            center_vec = obj_center[0] - obj_kp.pt[0], obj_center[1] - obj_kp.pt[1]
            if DRAW_VECTORS:
                cv2.line(object, (int(obj_kp.pt[0]), int(obj_kp.pt[1])),
                     (int(obj_kp.pt[0] + center_vec[0]), int(obj_kp.pt[1] + center_vec[1])), 255, 1)

            center_vec = (
                np.array(center_vec).dot(
                    cv2.getRotationMatrix2D(center=(0, 0), angle=delta_theta,
                                            scale=(self.scale_factor ** scale))[:, :2]))
            if DRAW_VECTORS:
                cv2.line(scene, (int(scene_kp.pt[0]), int(scene_kp.pt[1])),
                        (int(scene_kp.pt[0] + center_vec[0]), int(scene_kp.pt[1] + center_vec[1])), 255, 1)


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

        print(max_features, max_len)
        center = np.array([max_features[3] * self.scene_x_step+self.scene_x_step//2, max_features[2]* self.scene_y_step + self.scene_y_step//2])
        rect_kp = np.array([(-obj_W//2, -obj_H//2), (-obj_W//2, obj_H//2), (obj_W//2, obj_H//2), (obj_W//2, -obj_H//2)])
        rect_new = rect_kp.dot(cv2.getRotationMatrix2D(center=(0, 0), angle=(max_features[1]*self.angle_step + self.angle_step/2),
                                            scale=(self.scale_factor ** max_features[0]))[:, :2])
        rect_new = np.int0(rect_new + center)

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
            print(len(kps))
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
        print(x, rect_new, rect_kp)
        scene = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(scene, [rect_new], 0, (0, 255, 0), 2)
        img = cv2.drawMatches(object, kp1, scene, kp2, matches[:4], None)
        plt.imshow(img), plt.show()










if __name__ == "__main__":
    img_input = cv2.imread(OBJ_PATH)
    grey_obj = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    img_input = cv2.imread(SCENE_PATH)
    grey_scene = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    loc = Localizator()
    loc(grey_obj, grey_scene)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(grey_obj, None)
