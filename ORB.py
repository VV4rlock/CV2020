from config import *

from Harris_corner_detector import harris_response
from FAST_corner_detector import FAST
from BRIEF import RotatedBRIEF

class ORB:
    def __init__(self, R, bitsize):
        self.FAST_Radius = R
        self.OFAST_Radius = R
        self.FAST_threshold = 100
        self.get_descriptors = RotatedBRIEF(R, bitsize)
        self.Harris_threshold = 100
        self.scales = [1, 1/2, 1/4, 1/8]

    def similar(self, desc1, desc2):
        HW = (desc1 ^ desc2).sum()
        ones_similar = (desc1 & desc2).sum()
        zeros_similar = ((desc1 ^ 1) & (desc2 ^ 1)).sum()
        

if __name__ == "__main__":
    img_input = cv2.imread(IMAGE_PATH)
    grey_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY).astype(np.int16)