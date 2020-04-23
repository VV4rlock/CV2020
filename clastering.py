from config import *
from ORB import ORB
import re
from sklearn_extra.cluster import KMedoids

#r'Image filename : "VOC2005_1/PNGImages/Caltech_cars/image_0001.png"'
#r'Image filename : "VOC2005_1/PNGImages/UIUC_TestImages_Scale/test-80.png"'
filename_pattern = re.compile(r'Image filename : "VOC2005_\d/([/.\-_a-zA-Z0-9]*)"')
#r'Original label for object 1 "PAScarRear" : "carRear"'
obj_label_pattern = re.compile(r'Original label for object \d+ "(\w*)" : "(\w+)"')
#r'Bounding box for object 1 "PAScarRear" (Xmin, Ymin) - (Xmax, Ymax) : (41, 40) - (142, 113)'
#r'Bounding box for object 1 "PAScarRear" (Xmin, Ymin) - (Xmax, Ymax) : (410, 183) - (479, 242)'
bb_obj_pattern = re.compile(r'Bounding box for object \d+ "\w+" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((-?\d+), (-?\d+)\) - \((\d+), (\d+)\)')
orb = ORB(file_with_BRIEFpatch=BRIEF_PATCH)

DESCRIPTOR_DUMP_FILENAME = "descriptors_set.pickle"
def dump_descriptors_to_file(dump_filename):
    out = []
    names = []
    dirs = [os_path.join(VOC_PATH, "Annotations", i) for i in listdir(os_path.join(VOC_PATH, "Annotations"))]
    for dir in dirs:
        names += [os_path.join(dir, i) for i in listdir(dir)]
    #shuffle(names)
    names = names
    im_count = len(names)
    for im_index, name in enumerate(names):
        try:
            print(f"\rhandling {im_index}/{im_count}: {name}...", end='')
            with open(name) as f:
                data = f.read()
            image_filename = os_path.join(VOC_PATH, filename_pattern.findall(data)[0])
            all_objs_labels = obj_label_pattern.findall(data)
            objs = bb_obj_pattern.findall(data)
            assert len(all_objs_labels) == len(objs)
            orig_image = cv2.imread(image_filename)
            grey_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(grey_image)
            for index, labels in enumerate(all_objs_labels):
                if labels[1] in labels[0]:
                    x0, y0, x1, y1 = objs[index]
                    mask[int(y0):int(y1)+1, int(x0):int(x1)+1] = 255
            #show_image(grey_image)
            #show_image(mask)
            #orig_image = cv2.imread(image_filename)
            #grey_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
            angles, descriptors = orb(grey_image)
            for index, coords in enumerate(angles):
                if mask[coords[0], coords[1]] == 255:
                    out.append(descriptors[index])
            #show_image(orig_image)
        except Exception as e:
            logger.exception(e)
    print()
    import pickle
    with open(dump_filename, "wb") as f:
        pickle.dump(np.vstack(out), f)
    print(f"descriptors was dumped to {dump_filename}")

def k_medoids(k):
    import pickle
    with open(DESCRIPTOR_DUMP_FILENAME, 'rb') as f:
        descriptors = pickle.load(f)
    print(f"descriptors shape = {descriptors.shape}")
    kmedoids = KMedoids(n_clusters=5000, metric='hamming', init="heuristic", max_iter=300).fit(descriptors)
    print(kmedoids.cluster_centers_)
    print(f"inertial {kmedoids.inertia_}")





if __name__=="__main__":
    #dump_descriptors_to_file(DESCRIPTOR_DUMP_FILENAME)
    k_medoids(8)

