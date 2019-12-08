import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import numpy as np
import cv2
import torchvision
import myresnet as resnet
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from os import path
from PIL import Image
from sklearn.metrics import roc_curve, auc

# import multi_context
import copy

datafolder = 'part'
use_last_pretrained = False
# network_choice = 'InceptionResNetV2'
network_choice = 'resnet'
epoch_scratch = 200
training_progress = np.zeros((epoch_scratch, 5))
start_fold = 0
end_fold = 1
retrain_folds = np.asarray([6])
plt.ion()   # interactive mode
# network = ['ResNet-18', 'ResNet-34','ResNet-50', 'ResNet-101', 'ResNet-152']
network = ['ResNet-50']
# network = ['vgg']
# network = ['inception']

vec_mean = [177.68223469, 139.43425626, 179.30670566]
vec_std = [41.99594637, 51.20869017, 46.1423163]

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        # transforms.CenterCrop(32),
        # transforms.Resize(224),
        transforms.RandomResizedCrop(224, scale=(0.8, 1), ratio=(1.0,1.0)),
        # transforms.Resize(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(vec_mean, vec_std)
    ]),
    'val': transforms.Compose([
        # transforms.CenterCrop(32),
        # transforms.Resize(224),
        transforms.RandomResizedCrop(224, scale=(0.9, 1), ratio=(1.0,1.0)),
        # transforms.Resize(320),
        transforms.ToTensor(),
        # transforms.Normalize(vec_mean, vec_std)
    ]),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(vec_mean)
    std = np.array(vec_std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
#inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
#out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])

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
        # print('classes\n{}'.format(classes))
        # print('class_to_idx\n{}'.format(class_to_idx))
        # time.sleep(30)
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
        # print('target {}'.format(target))
        # print('img_path {}'.format(img_path))
        # image = cv2.imread(img_path)
        # cv2.imshow('show', image)
        # cv2.waitKey(0)
        # print('img_path {}'.format(img_path))
        # time.sleep(30)
        # image = (image - np.mean(image)) / np.std(image)
        # image = Image.fromarray(image)
        with open(img_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        # print('='*30)
        image = (image - torch.mean(image)) / torch.std(image)
        # print(image.shape)
        # time.sleep(30)
        return image, target, img_id

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

def train_model(model, criterion, optimizer, scheduler, fn_save, num_epochs=25):
    since = time.time()

    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_ep = 0
    best_auc = 0.0
    test_scores = []
    test_labels = []
    tv_hist = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, img_id in dataloaders[phase]:
                # Get images from inputs
                #print('*'*10 + ' printing inputs and labels ' + '*'*10)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print('labels {}'.format(labels))
                # print('inputs shape {}'.format(inputs.shape))

                # print(img_id)
                # print(manual_scores)
                # print(agatston_scores)
                # print(labels)
                # time.sleep(10)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # print('{}'.format(list.__sizeof__()))
                    # outputs = model(inputs)
                    # outputs = model(input_patch)
                    # print('labels\n{}'.format(labels))
                    # time.sleep(30)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # print('let me check################################################')
                    probs = nn.functional.softmax(outputs, dim=1)
                    # print(probs[0])
                    # print(labels[0])
                    # print(torch.sum(probs[0]))
                    # time.sleep(30)

                    loss = criterion(probs, labels)
                    # print(loss)
                    # time.sleep(30)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        scores = nn.functional.softmax(outputs, dim=1)
                        test_scores.extend(scores.data.cpu().numpy()[:, 1])
                        test_labels.extend(labels.data.cpu().numpy())


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            tv_hist[phase].append([epoch_loss, epoch_acc])

            ''' Calculate this round of AUC score '''
            # epoch_auc = 0.0
            # if phase == 'val' and len(test_scores) != 0:
            #     # print('test_labels {}, test_scores {}'.format(test_labels.shape, test_scores.shape))
            #     fpr, tpr, _ = roc_curve(test_labels, test_scores)
            #     epoch_auc = auc(fpr, tpr)
            #     if epoch_auc < 0.5:
            #         test_scores = np.asarray(test_scores)
            #         test_scores = np.ones_like(test_scores) - test_scores
            #         test_scores = test_scores.tolist()
            #         # print('test_labels {}, test_scores {}'.format(test_labels.shape, test_scores.shape))
            #         # time.sleep(30)
            #         fpr, tpr, _ = roc_curve(test_labels, test_scores)
            #         epoch_auc = auc(fpr, tpr)
            #     print('roc_auc {:.4f}'.format(epoch_auc))

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
            # if phase == 'val' and epoch_auc >= best_auc:
            #     best_auc = epoch_auc
                best_acc = epoch_acc
                best_ep = epoch
                #best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), fn_save)
                # print('**** best model updated with auc={:.4f} ****'.format(epoch_auc))
                print('**** best model updated with acc={:.4f} ****'.format(epoch_acc))


        print('ep {}/{} - Train loss: {:.4f} acc: {:.4f}, Val loss: {:.4f} acc: {:.4f}'.format(
            epoch + 1, num_epochs, 
            tv_hist['train'][-1][0], tv_hist['train'][-1][1],
            tv_hist['val'][-1][0], tv_hist['val'][-1][1]))
        training_progress[epoch][0] = tv_hist['train'][-1][0]
        training_progress[epoch][1] = tv_hist['train'][-1][1]
        training_progress[epoch][2] = tv_hist['val'][-1][0]
        training_progress[epoch][3] = tv_hist['val'][-1][1]
        training_progress[epoch][4] = 0
        np.savetxt('training_progress.txt', training_progress)


        #print('-' * 10)

        #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        #    phase, epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('*'*10 + 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('*'*10 + 'Best val Acc: {:4f} at epoch {}'.format(best_acc, best_ep))
    print()

    # load best model weights
    #model.load_state_dict(best_model_wts)
    #return model
    return tv_hist


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


#
# %% 10-fold cross validation
#

k_tot = 10

for net in network:
    epoch_ft = 100
    epoch_conv = 100
    # epoch_scratch = 150
    if net == 'ResNet-18':
        base_model = resnet.resnet18
        #continue
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
    else:
        print('The network of {} is not supported!'.format(net))

    data_dir = path.expanduser('data')
    # data_dir = path.expanduser('~/tmp/dog_data/small_set')
    image_datasets = {x: MortalityRiskDataset(os.path.join(data_dir, x),
                                              data_transforms[x])
                    for x in ['train', 'val']}
    print('image_datasets\n{}'.format(image_datasets))
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128,
                                                shuffle=True, num_workers=0)
                for x in ['train', 'val']}
    print('dataloaders\n{}'.format(dataloaders))
    print('size of dataloader: {}'.format(dataloaders.__sizeof__()))
    # time.sleep(30)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    ######################################################################
    # Train from scratch



    if network_choice == 'InceptionResNetV2':
        model_ft = InceptionResNetV2(num_classes=2)
        model_ft = model_ft.to(device)
    elif network_choice == 'vgg':
        model_ft = vgg.vgg19(pretrained=False)
        model_ft.cuda()
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, 2)
        model_ft = model_ft.to(device)
    elif network_choice == 'inception':
        model_ft = inception.InceptionA()
        model_ft.cuda()
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, 2)
        model_ft = model_ft.to(device)
    else:
        model_ft = base_model(pretrained=True)
        # model_ft = base_model(pretrained=True)
        # model_ft.cuda()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 120)
        model_ft = model_ft.to(device)

    if use_last_pretrained:
        fn_best_model = os.path.join(data_dir, 'best_scratch_{}.pth'.format(net))
        model_ft.load_state_dict(torch.load(fn_best_model))
        model_ft.eval()
        # model_ft.cuda()

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.L1Loss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-5)
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.9)

    # Train and evaluate
    fn_best_model = os.path.join(data_dir, 'best_scratch_{}.pth'.format(net))
    hist_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        fn_best_model, num_epochs=epoch_scratch)
    fn_hist = os.path.join(data_dir, 'hist_scratch_{}.npy'.format(net))
    np.save(fn_hist, hist_ft)
    txt_path = path.join(data_dir, 'training_progress_{}.txt'.format(net))
    np.savetxt(txt_path, training_progress)
    print('h#' * 30)
    # break
    ######################################################################
