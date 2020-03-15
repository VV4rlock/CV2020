from config import *

def BRIEF(image: np.ndarray):
    pass

if __name__=="__main__":
    img_input = cv2.imread(IMAGE_PATH)
    BRIEF(img_input)