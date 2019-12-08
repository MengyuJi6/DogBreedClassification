import cv2
import torch
import torch.nn as nn
import myresnet as resnet
from torch.utils.data import Dataset
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from os import path
from PIL import Image
import pandas as pd
import kaggle_ui
######################################


# datafolder = 'all'
datafolder = 'part'
# datafolder = 'submission_test'
start_fold = 0
end_fold = 1
vec_mean = [177.68223469, 139.43425626, 179.30670566]
vec_std = [41.99594637, 51.20869017, 46.1423163]
container = np.ones((1, 2))
batch_size = 8

# Just normalization for validation
data_transform = transforms.Compose([
        # transforms.Resize(int(224*1.5)),
        # transforms.Resize(320),
        # transforms.Resize(320),
        # transforms.CenterCrop(224),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(vec_mean, vec_std)
    ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# network = ['ResNet-101', 'ResNet-152']
network = ['ResNet-50']

# outfile = open('acc_results/fc'+network[0] + '.txt', 'a+')

# input an image array
# normalize values to 0-255
def array_normalize(input_array):
    max_value = np.max(input_array)
    min_value = np.min(input_array)
    # print('max {}, min {}'.format(max_value, min_value))
    k = 255 / (max_value - min_value)
    min_array = np.ones_like(input_array) * min_value
    normalized = k * (input_array - min_array)
    return normalized

# input the weights of neurons of fc layers
# output the normalized weights
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1) # only difference

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

class MortalityRiskDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        """
        classes, class_to_idx = find_classes(root_dir)
        samples = make_dataset(root_dir, class_to_idx, IMG_EXTENSIONS)

        self.root_dir = root_dir
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        img_path, target = self.samples[idx]
        img_id = img_path[-44: -4]

        # 0a6d8d83e9a57cc0bf31a2749af2582a21b56cab

        with open(img_path, 'rb') as f:
            print('f {}'.format(f))
            time.sleep(30)
            image = Image.open(f)
            image = image.convert('RGB')

        cv2.imshow('image', image)
        cv2.waitKey(0)

        if self.transform:
            image = self.transform(image)

        return image, target, img_id

def tif_to_tensor(img_path):
    # img_path = kaggle_ui.img_path
    # global kaggle_ui.img_path
    with open(img_path, 'rb') as f:
        # print('f {}'.format(f))
        # time.sleep(30)
        image = Image.open(f)
        image = image.convert('RGB')

    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    image = data_transform(image)
    image_np = image.data.cpu().numpy()
    image_np = np.reshape(image_np, (1, image_np.shape[0], image_np.shape[1], image_np.shape[2]))
    image = torch.from_numpy(image_np).float().to(device)

    print('transferred {}'.format(image.shape))
    return image

def load_model(modelName):
    net = modelName[-13:-4]
    print('net {}'.format(net))
    model_x = resnet.resnet50()
    num_ftrs = model_x.fc.in_features

    model_x.fc = nn.Linear(num_ftrs, 2)
    # model_x.fc = nn.Linear(513, 2)

    # data_dir = path.expanduser('~/tmp/my_cross_val_10/fold_0')
    # fn_best_model = os.path.join(data_dir, 'best_{}_{}.pth'.format(mode, net))
    fn_best_model = modelName
    # print(fn_best_model)
    model_x.load_state_dict(torch.load(fn_best_model, map_location=lambda storage, loc: storage))
    model_x.eval()

    model_x = model_x.to(device)
    print('Trained model loaded successfully!')
    return model_x

def save_slices(inputs, list, preds):
    x_numpy = np.asarray(inputs)
    preds_numpy = np.asarray(preds)
    for i in range(0, x_numpy.shape[0]):
        image = x_numpy[i, :, :, :]
        image_c0 = np.reshape(image[0, :, :], (image.shape[1], image.shape[2], 1))
        image_c1 = np.reshape(image[1, :, :], (image.shape[1], image.shape[2], 1))
        image_c2 = np.reshape(image[2, :, :], (image.shape[1], image.shape[2], 1))
        image = np.concatenate((image_c2, image_c1), axis=2)
        image = np.concatenate((image, image_c0), axis=2)
        image = array_normalize(image)
        cv2.imwrite('experiments/heatmaps_{}/{}_L{}P{}_{}.jpg'.format(training_part, list[4][i], list[3][i],
                                                                   preds_numpy[i], training_part), image)

def save_heatmap(featuremap, img_id, class_id):
    plt.imsave('experiments/heatmaps_{}/{}_P{}_{}.jpg'.format(training_part, img_id, class_id, training_part), featuremap,
               cmap='jet_r')

def test_model(model):
    '''Test the trained models'''

    since = time.time()

    test_scores = []
    test_labels = []
    running_corrects = 0
    img_index = 1
    columns = ['id', 'label']
    pd_table = pd.DataFrame(columns=columns)

    # Iterate over data.
    for inputs, labels, img_id in dataloader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        scores = nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        preds_np = preds.data.cpu().numpy()

        for i in range(preds_np.shape[0]):
            img_pid = img_id[i]
            img_preds = preds_np[i]
            pd_table = pd_table.append({'id': img_pid, 'label': img_preds}, ignore_index=True)
            print('{}/57458: id {}, preds {}'.format(img_index, img_pid, img_preds))
            img_index += 1
            # time.sleep(1)
        pd_table.to_csv('submissions/0405_40epochs.csv')
        # time.sleep(30)

        test_scores.extend(scores.data.cpu().numpy()[:, 1])
        test_labels.extend(labels.data.cpu().numpy())

        running_corrects += torch.sum(preds == labels.data)
    acc = running_corrects.double() / dataset_size

    # outfile.write('{:.4f} '.format(acc))
    print('Test acc: {:.4f}'.format(acc))
    # file = open('data/results.txt', 'a')
    # file.write(acc)
    # file.close()

    time_elapsed = time.time() - since
    return test_scores, test_labels, acc


if __name__ == '__main__':
    k_tot = 10
    print(os.getcwd())
    avg_acc = 0
    #time.sleep(10)

    for net in network:
        if net == 'ResNet-18':
            base_model = resnet.resnet18
        elif net == 'ResNet-34':
            base_model = resnet.resnet34
            #continue
        elif net == 'ResNet-50':
            base_model = resnet.resnet50
            #continue
        elif net == 'ResNet-101':
            base_model = resnet.resnet101
            #continue
        elif net == 'ResNet-152':
            base_model = resnet.resnet152
            #continue
        else:
            print('The network of {} is not supported!'.format(net))

        #file.write('safdsaf')
        # for k in range(k_tot):
        for k in range(start_fold, end_fold):
            print('Testing fold {}/{} of {}'.format(k+1, k_tot, net))
            #file.write('Testing fold {}/{} of {}'.format(k+1, k_tot, net))
            #print(os.getcwd())
            data_dir = path.expanduser('/zion/guoh9/kaggle/cancer/{}'.format(datafolder))
            #image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
            #                                         data_transform)
            image_dataset = MortalityRiskDataset(os.path.join(data_dir, 'val'),
                                                      data_transform)

            dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size,
                                                         shuffle=False, num_workers=0)

            dataset_size = len(image_dataset)

            class_names = image_dataset.classes

            #entry = [net]
            #print(entry)
            #print('what is entry')
            ft_acu = 0
            conv_acu = 0
            scratch_acu = 0

            #for mode in ['ft', 'conv', 'scratch']:
            for mode in ['scratch']:
                model_x = base_model(pretrained=False)
                num_ftrs = model_x.fc.in_features

                model_x.fc = nn.Linear(num_ftrs, 2)

                fn_best_model = os.path.join(data_dir, 'best_{}_{}.pth'.format(mode, net))
                # print(fn_best_model)
                model_x.load_state_dict(torch.load(fn_best_model, map_location=lambda storage, loc: storage))
                model_x.eval()

                model_x = model_x.to(device)
                # print(model_x)
                # time.sleep(30)
                print(mode+': ', end='')
                scores, labels, myacc = test_model(model_x)
                avg_acc = avg_acc + myacc
                #entry.append(str(myacc))


                results = np.asarray([scores, labels])
                fn_results = os.path.join(data_dir, 'test_results_{}_{}.npy'.format(mode, net))
                np.save(fn_results, results)
                # print(scores.shape)
                # print(labels.shape)
    print('avg_acc = {:.4f}'.format(avg_acc/10))
