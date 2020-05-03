from config import *
from ORB import ORB
import re
from sklearn_extra.cluster import KMedoids
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid


#r'Image filename : "VOC2005_1/PNGImages/Caltech_cars/image_0001.png"'
#r'Image filename : "VOC2005_1/PNGImages/UIUC_TestImages_Scale/test-80.png"'
filename_pattern = re.compile(r'Image filename : "VOC2005_\d/([/.\-_a-zA-Z0-9]*)"')
#r'Original label for object 1 "PAScarRear" : "carRear"'
obj_label_pattern = re.compile(r'Original label for object \d+ "(\w*)" : "(\w+)"')
#r'Bounding box for object 1 "PAScarRear" (Xmin, Ymin) - (Xmax, Ymax) : (41, 40) - (142, 113)'
#r'Bounding box for object 1 "PAScarRear" (Xmin, Ymin) - (Xmax, Ymax) : (410, 183) - (479, 242)'
bb_obj_pattern = re.compile(r'Bounding box for object \d+ "\w+" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((-?\d+), (-?\d+)\) - \((\d+), (\d+)\)')


CLASSES = ['car', 'person', 'bike', 'bicycle']
label_matcher = re.compile(rf'.*({"|".join(CLASSES)})+.*')
#orb = ORB(file_with_BRIEFpatch=BRIEF_PATCH)
clastering_orb = cv2.ORB_create(nfeatures=50)
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

DESCRIPTOR_DUMP_FILENAME = "descriptors_set.pickle"
def dump_descriptors_to_file(dump_filename):
    out = {i: [] for i in CLASSES}
    names = []
    dirs = [os_path.join(VOC_TRAIN_PATH, "Annotations", i) for i in listdir(os_path.join(VOC_TRAIN_PATH, "Annotations"))]
    for dir in dirs:
        names += [os_path.join(dir, i) for i in listdir(dir)]
    shuffle(names)
    names = names
    im_count = len(names)
    for im_index, name in enumerate(names):
        try:
            print(f"\rhandling {im_index}/{im_count}: {name}...", end='')
            with open(name) as f:
                data = f.read()
            image_filename = os_path.join(VOC_TRAIN_PATH, filename_pattern.findall(data)[0])
            all_objs_labels = obj_label_pattern.findall(data)

            objs = bb_obj_pattern.findall(data)
            assert len(all_objs_labels) == len(objs)
            orig_image = cv2.imread(image_filename)
            grey_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(grey_image)
            for index, labels in enumerate(all_objs_labels):
                if label_matcher.match(labels[0]):
                    x0, y0, x1, y1 = list(map(lambda x: max(x,0), map(int, objs[index])))
                    if x0 > x1:
                        x0, x1 = x1, x0
                    if y0 > y1:
                        y0, y1 = y1, y0
                    mask[int(y0):int(y1)+1, int(x0):int(x1)+1] = 255

                    kp, descriptors = clastering_orb.detectAndCompute(grey_image, mask)

                    if kp:
                        for _class in CLASSES:
                            if _class in labels[0]:
                                out[_class].append(descriptors)
                    else:
                        continue # comment for view errors
                        img2 = cv2.drawKeypoints(grey_image, kp, None, color=(0, 255, 0), flags=0)
                        img2[:, :, 2] = mask
                        print(all_objs_labels, objs)
                        print(data)
                        plt.imshow(img2), plt.show()

            #show_image(orig_image)
        except Exception as e:
            logger.exception(e)
    print("\nDescriptor counts:")
    for _class in CLASSES:
        descriptors[_class] = np.vstack(descriptors[_class])
        print(f"\t{_class}: {descriptors[_class].shape[0]}")

    import pickle
    with open(dump_filename, "wb") as f:
        pickle.dump(out, f)
    print(f"descriptors was dumped to {dump_filename}")


ONE_CLASS_DESCRIPTOR_COUNT = 9000
NROF_CLASTERS = 5000
ALL = False
THRESHOLD = 100 # близость к медоиду
def k_medoids():
    import pickle
    with open(DESCRIPTOR_DUMP_FILENAME, 'rb') as f:
        descriptors = pickle.load(f)
    print("\nDescriptor counts:")
    for _class in CLASSES:
        print(f"\t{_class}: {descriptors[_class].shape[0]}")


    claster_desctiptors = []
    for _class in CLASSES:
        if ALL:
            claster_desctiptors.append(descriptors[_class])
        else:
            claster_desctiptors.append(descriptors[_class]\
                                       [np.random.choice(descriptors[_class].shape[0], size=ONE_CLASS_DESCRIPTOR_COUNT
                                                         , replace=False), :])

    claster_desctiptors = np.unpackbits(np.vstack(claster_desctiptors), axis=1)
    print(f"descriptors shape for clastering = {claster_desctiptors.shape}")

    kmedoids_obj = KMedoids(n_clusters=NROF_CLASTERS, metric='hamming', init="k-medoids++", max_iter=300).fit(claster_desctiptors)
    return kmedoids_obj
'''
    claster_objs = []
    claster_per_class = NROF_CLASTERS // (len(CLASSES))
    for _class in CLASSES:

        #claster_desctiptors.append(descriptors[_class])

        claster_desctiptors = np.unpackbits(descriptors[_class], axis=1)
        print(f"descriptors shape for clastering = {claster_desctiptors.shape}")

        kmedoids_obj = KMedoids(n_clusters=claster_per_class, metric='hamming', init="k-medoids++", max_iter=300).fit(
            claster_desctiptors)
        claster_objs.append(kmedoids_obj)


    return claster_objs
'''

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(NROF_CLASTERS, 100),
            nn.ReLU(),
            nn.Linear(100, len(CLASSES))
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def term_frequency_encoding(descriptors, medoids_obj):

    dintandces = medoids_obj.transform(descriptors)
    indices = np.argmin(dintandces, axis=1)
    clasters_on_image = indices[dintandces[np.arange(dintandces.shape[0]), indices] < THRESHOLD]
    res = np.zeros(NROF_CLASTERS)
    np.add.at(res, clasters_on_image, 1)
    return res
'''
    out = []
    for med in medoids_obj:
        dintandces = med.transform(descriptors)
        indices = np.argmin(dintandces, axis=1)
        clasters_on_image = indices[dintandces[np.arange(dintandces.shape[0]), indices] < THRESHOLD]
        res = np.zeros(NROF_CLASTERS // len(CLASSES))
        np.add.at(res, clasters_on_image, 1)
        out.append(res)
    out = np.concatenate(out)
    return out
'''

def binary_encoding(descriprors, medoids_obj):
    res = term_frequency_encoding(descriprors, medoids_obj)
    res[res > 0] = 1
    return res

class VOCDataset(Dataset):
    def __init__(self,voc_path, medoids_object, encoding_func):
        dirs_by_class = {i: [] for i in CLASSES}
        self.samples = []
        temp_y_vector = np.zeros(len(CLASSES))
        dirs = [os_path.join(voc_path, "PNGImages", i) for i in listdir(os_path.join(voc_path, "PNGImages"))]

        for dir in dirs:
            for _class in CLASSES:
                if _class in dir:
                    dirs_by_class[_class].append(dir)

        for index, _class in enumerate(CLASSES):
            y_t = temp_y_vector[:]
            y_t[index] = 1
            for dir in dirs_by_class[_class]:
                for image_filename in listdir(dir):
                    image = cv2.imread(os_path.join(dir, image_filename))
                    kp, descriptors = clastering_orb.detectAndCompute(image, None)
                    x_t = encoding_func(np.unpackbits(descriptors, axis=1), medoids_object)
                    self.samples.append((torch.tensor(x_t).float(), index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]



def learn_MLP(medoids_obj):
    train_dataset = VOCDataset(VOC_TRAIN_PATH, medoids_obj, term_frequency_encoding)
    test_dataset = VOCDataset(VOC_TEST_PATH, medoids_obj, term_frequency_encoding)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    model = MLP()
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    mean_train_losses = []
    mean_valid_losses = []
    valid_acc_list = []
    epochs = 15

    for epoch in range(epochs):
        model.train()

        train_losses = []
        valid_losses = []
        for i, (images, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                valid_losses.append(loss.item())

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        mean_train_losses.append(np.mean(train_losses))
        mean_valid_losses.append(np.mean(valid_losses))

        accuracy = 100 * correct / total
        valid_acc_list.append(accuracy)
        print('epoch : {}, train loss : {:.4f}, test loss : {:.4f}, test acc : {:.2f}%' \
              .format(epoch + 1, np.mean(train_losses), np.mean(valid_losses), accuracy))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    ax1.plot(mean_train_losses, label='train')
    ax1.plot(mean_valid_losses, label='valid')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='best')

    ax2.plot(valid_acc_list, label='valid acc')
    ax2.legend()
    plt.show()



if __name__=="__main__":
    #dump_descriptors_to_file(DESCRIPTOR_DUMP_FILENAME)
    medoid_obj = k_medoids()
    learn_MLP(medoid_obj)


