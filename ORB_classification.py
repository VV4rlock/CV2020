from config import *
from ORB import ORB
import re
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle

def term_frequency_encoding(descriptors, claster_obj):
    descriptors = np.unpackbits(descriptors, axis=1)
    dintandces = claster_obj.transform(descriptors)
    indices = np.argmin(dintandces, axis=1)
    clasters_on_image = indices[dintandces[np.arange(dintandces.shape[0]), indices] < MEDOIDS_DISTANCE_TO_INCLUDE]
    res = np.zeros(NROF_CLASTERS)
    np.add.at(res, clasters_on_image, 1)
    return res

def binary_encoding(descriprors, claster_obj):
    res = term_frequency_encoding(descriprors, claster_obj)
    res[res > 0] = 1
    return res


CLASSES = ['car', 'person', 'bike', 'bicycle']
filename_pattern = re.compile(r'Image filename : "VOC2005_\d/([/.\-_a-zA-Z0-9]*)"')
obj_label_pattern = re.compile(r'Original label for object \d+ "(\w*)" : "(\w+)"')
bb_obj_pattern = re.compile(r'Bounding box for object \d+ "\w+" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((-?\d+), (-?\d+)\) - \((\d+), (\d+)\)')
label_matcher = re.compile(rf'.*({"|".join(CLASSES)})+.*')


ENCODING_FUNC = binary_encoding
ONE_CLASS_DESCRIPTOR_COUNT = 8000
NROF_CLASTERS = 1500
USE_ALL_DESCRIPTORS_FOR_CLASTERING = False
GENERATE_DESCRIPTORS = False
MEDOIDS_DISTANCE_TO_INCLUDE = 3 # близость к медоиду
USE_CLASTERING = False
DUMP_DATASETS_TO_FILE = False
clastering_orb = cv2.ORB_create()
orb = cv2.ORB_create()

DESCRIPTOR_DUMP_FILENAME = "descriptors_set.pickle"
MASK_ONLY = False
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
            if MASK_ONLY:
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
            else:
                kp, descriptors = clastering_orb.detectAndCompute(grey_image, None)

                if kp:
                    out[CLASSES[np.random.randint(len(CLASSES))]].append(descriptors)

            #show_image(orig_image)
        except Exception as e:
            logger.exception(e)
    print("\nDescriptor counts:")
    for _class in CLASSES:
        out[_class] = np.vstack(out[_class])
        print(f"\t{_class}: {out[_class].shape[0]}")

    import pickle
    with open(dump_filename, "wb") as f:
        pickle.dump(out, f)
    print(f"descriptors was dumped to {dump_filename}")


def k_medoids():
    import pickle
    with open(DESCRIPTOR_DUMP_FILENAME, 'rb') as f:
        descriptors = pickle.load(f)
    print("\nDescriptors counts:")
    for _class in CLASSES:
        print(f"\t{_class}: {descriptors[_class].shape[0]}")


    claster_desctiptors = []
    for _class in CLASSES:
        if USE_ALL_DESCRIPTORS_FOR_CLASTERING:
            claster_desctiptors.append(descriptors[_class])
        else:
            claster_desctiptors.append(descriptors[_class]\
                                       [np.random.choice(descriptors[_class].shape[0], size=ONE_CLASS_DESCRIPTOR_COUNT
                                                         , replace=False), :])

    claster_desctiptors = np.unpackbits(np.vstack(claster_desctiptors), axis=1)
    print(f"descriptors shape for clastering = {claster_desctiptors.shape}")

    kmedoids_obj = KMedoids(n_clusters=NROF_CLASTERS, metric='hamming', init="k-medoids++", max_iter=300).fit(claster_desctiptors)
    return kmedoids_obj


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        hidden = 10
        self.layers = nn.Sequential(
            nn.Linear(NROF_CLASTERS, hidden),
            nn.ReLU(),
            nn.Linear(hidden, len(CLASSES))
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class VOCDataset(Dataset):
    def __init__(self, voc_path, medoids_object, encoding_func, inverted_doc_fq=False, is_train=True):
        self.samples = []
        self.labels = []

        if medoids_object is None:
            file = "vectorsVOC2005_{}.pickle".format("1" if is_train else "2")
            with open(file, 'rb') as f:
                self.samples, self.labels = pickle.load(f)
            return

        names = []
        dirs = [os_path.join(voc_path, "Annotations", i) for i in
                listdir(os_path.join(voc_path, "Annotations"))]
        for dir in dirs:
            names += [os_path.join(dir, i) for i in listdir(dir)]

        #names = names
        termfq = np.zeros(NROF_CLASTERS)
        im_count = len(names)
        for im_index, name in enumerate(names):
            try:
                print(f"\rhandling {im_index}/{im_count}: {name}...", end='')
                with open(name) as f:
                    data = f.read()
                image_filename = os_path.join(voc_path, filename_pattern.findall(data)[0])
                all_objs_labels = obj_label_pattern.findall(data)

                orig_image = cv2.imread(image_filename)
                grey_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)

                classes = []
                for index, labels in enumerate(all_objs_labels):
                    for _class in CLASSES:
                        if _class in labels[0] and _class not in classes:
                            classes.append(_class)

                if len(classes) == 1:
                    kp, descriptors = clastering_orb.detectAndCompute(grey_image, None)

                    if kp:
                        x = encoding_func(descriptors, medoids_object)
                        self.samples.append(torch.tensor(x).float())
                        self.labels.append(CLASSES.index(classes[0]))
                        termfq[x > 0] += 1
            except Exception as e:
                logger.exception(e)

        print(termfq)
        termfq = np.log(len(names)/termfq)

        if inverted_doc_fq:
            for index in range(len(self.samples)):
                self.samples[index] = self.samples[index]*termfq

        if DUMP_DATASETS_TO_FILE:
            file = "vectorsVOC2005_{}.pickle".format("1" if is_train else "2")
            with open(file, "wb") as f:
                pickle.dump((self.samples, self.labels), f)

        print(f"VOC SIZE of {voc_path} is {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]


def mAP(correct, outs, labels):
    s = 0
    for k in range(len(CLASSES)):
        #indices = np.where(labels == k)
        s += average_precision_score(correct, outs[:, k])
        #print(f"{k}: {s}")
    return s/len(CLASSES)



def learn_MLP(medoids_obj, encoding_func):
    batchsize = 32
    train_dataset = VOCDataset(VOC_TRAIN_PATH, medoids_obj, encoding_func, is_train=True)
    test_dataset = VOCDataset(VOC_TEST_PATH, medoids_obj, encoding_func, is_train=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=False)

    model = MLP()
    print(model)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    mean_train_losses = []
    mean_test_losses = []
    test_acc_list = []
    train_acc_list = []
    epochs = 15
    softmax = torch.nn.Softmax(dim=1)

    for epoch in range(epochs):

        model.train()

        train_losses = []
        test_losses = []
        correct = 0
        total = 0
        outs = []
        labels_vec = []
        correct_vec = []

        for i, (images, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            outs.append(softmax(outputs.data).numpy())
            labels_vec.append(labels.data.numpy())
            correct_vec.append((predicted == labels).data.numpy())

        train_accuracy = 100 * correct / total
        train_acc_list.append(train_accuracy)

        outs = np.vstack(outs)
        labels_vec = np.concatenate(labels_vec)
        correct_vec = np.concatenate(correct_vec)
        train_map = mAP(correct_vec, outs, labels_vec)

        model.eval()
        correct = 0
        total = 0
        outs = []
        labels_vec = []
        correct_vec = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                test_losses.append(loss.item())

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                outs.append(softmax(outputs.data).numpy())
                labels_vec.append(labels.data.numpy())
                correct_vec.append((predicted == labels).data.numpy())

        mean_train_losses.append(np.mean(train_losses))
        mean_test_losses.append(np.mean(test_losses))

        outs = np.vstack(outs)
        labels_vec = np.concatenate(labels_vec)
        correct_vec = np.concatenate(correct_vec)
        test_map = mAP(correct_vec, outs, labels_vec)

        accuracy = 100 * correct / total
        test_acc_list.append(accuracy)
        print('epoch : {}, train/test loss : ({:.4f} / {:.4f}) acc : ({:.2f}% / {:.2f}%) map: ({:.4f} / {:.4f}) '\
              .format(epoch + 1, np.mean(train_losses), np.mean(test_losses), train_accuracy, accuracy, train_map, test_map))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    ax1.plot(mean_train_losses, label='train')
    ax1.plot(mean_test_losses, label='test')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='best')

    ax2.plot(test_acc_list, label='test acc')
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    if GENERATE_DESCRIPTORS:
        dump_descriptors_to_file(DESCRIPTOR_DUMP_FILENAME)
    if USE_CLASTERING:
        medoid_obj = k_medoids()
    else:
        medoid_obj = None
    learn_MLP(medoid_obj, ENCODING_FUNC)


