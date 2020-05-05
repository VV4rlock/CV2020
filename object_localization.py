from config import *
from matplotlib import pyplot as plt

OBJ_PATH = r'box.png'
SCENE_PATH = r'box_in_scene.png'


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
        print(scene_H, scene_W)
        for match in matches:
            obj_kp = kp1[match.queryIdx]
            scene_kp = kp2[match.trainIdx]
            delta_theta = scene_kp.angle - obj_kp.angle
            scale = scene_kp.octave - obj_kp.octave
            center_vec = obj_center[0] - obj_kp.pt[0], obj_center[1] - obj_kp.pt[1]
            cv2.line(object, (int(obj_kp.pt[0]), int(obj_kp.pt[1])),
                     (int(obj_kp.pt[0] + center_vec[0]), int(obj_kp.pt[1] + center_vec[1])), 255, 1)
            center_vec = (
                np.array(center_vec).dot(
                    cv2.getRotationMatrix2D(center=(0, 0), angle=delta_theta,
                                            scale=(self.scale_factor ** scale))[:, :2]))
            cv2.line(scene, (int(scene_kp.pt[0]), int(scene_kp.pt[1])),
                     (int(scene_kp.pt[0] + center_vec[0]), int(scene_kp.pt[1] + center_vec[1])), 255, 1)


            features = (scale, int(delta_theta // self.angle_step), int((scene_kp.pt[1] + center_vec[1]) // self.scene_y_step),
                        int((scene_kp.pt[0] + center_vec[0]) // self.scene_x_step))
            print(features)
            try:
                curr_list = accum[features[0]][features[1]][features[2]][features[3]]
                # noinspection PyTypeChecker
                curr_list.append((obj_kp, scene_kp))
                if len(curr_list) > max_len:
                    max_keypoints = curr_list
                    max_features = features
                    max_len = len(curr_list)
            except IndexError as e:
                continue # bad match?? point + center_vec

        print(max_features, max_len)
        center = np.array([max_features[3] * self.scene_x_step+self.scene_x_step//2, max_features[2]* self.scene_y_step + self.scene_y_step//2])
        rect_kp = np.array([(-obj_W//2, -obj_H//2), (-obj_W//2, obj_H//2), (obj_W//2, obj_H//2), (obj_W//2, -obj_H//2)])
        rect_new = rect_kp.dot(cv2.getRotationMatrix2D(center=(0, 0), angle=(max_features[1]*self.angle_step + self.angle_step/2),
                                            scale=(self.scale_factor ** max_features[0]))[:, :2])
        rect_new = np.int0(rect_new + center)
        
        scene = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(scene, [rect_new], 0, (0, 255, 0), 2)



        img = cv2.drawMatches(object, kp1, scene, kp2, matches, None)
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
