from config import *
from matplotlib import pyplot as plt

OBJ_PATH = r'box.png'
SCENE_PATH = r'box_in_scene.png'

class Localizator():
    def __init__(self):
        self.orb = cv2.ORB_create()

    def __call__(self, object, scene):
        kp1, des1 = self.orb.detectAndCompute(object, None)
        kp2, des2 = self.orb.detectAndCompute(scene, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        match_img = cv2.drawMatches(object, kp1, scene, kp2, matches[:10], None)
        plt.imshow(match_img), plt.show()


if __name__ == "__main__":
    img_input = cv2.imread(OBJ_PATH)
    grey_obj = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    img_input = cv2.imread(SCENE_PATH)
    grey_scene = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    loc = Localizator()
    loc(grey_obj, grey_scene)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(grey_obj, None)


