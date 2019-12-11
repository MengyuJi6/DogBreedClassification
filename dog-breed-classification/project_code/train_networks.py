import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import numpy as np
import myresnet as resnet
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from os import path
from PIL import Image

datafolder = 'part'
use_last_pretrained = False
network_choice = 'resnet'
epoch_scratch = 200
training_progress = np.zeros((epoch_scratch, 5))
start_fold = 0
end_fold = 1
retrain_folds = np.asarray([6])
plt.ion()   # interactive mode
network = ['ResNet-50']
# network = ['vgg']

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1), ratio=(1.0,1.0)),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1), ratio=(1.0,1.0)),
        transforms.ToTensor(),
    ]),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        with open(img_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        image = (image - torch.mean(image)) / torch.std(image)

        return image, target, img_id

######################################################################
# Training the model
######################################################################

def train_model(model, criterion, optimizer, scheduler, fn_save, num_epochs=25):
    since = time.time()
    best_acc = 0.0
    best_ep = 0
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
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    probs = nn.functional.softmax(outputs, dim=1)

                    loss = criterion(probs, labels)

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

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_ep = epoch
                torch.save(model.state_dict(), fn_save)
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

    time_elapsed = time.time() - since
    print('*'*10 + 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('*'*10 + 'Best val Acc: {:4f} at epoch {}'.format(best_acc, best_ep))
    print()

    return tv_hist

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
        model_ft = vgg.vgg16(pretrained=False)
        model_ft.cuda()
        model_ft = model_ft.to(device)
    elif network_choice == 'inception':
        model_ft = inception.InceptionA()
        model_ft.cuda()
        model_ft = model_ft.to(device)
    else:
        model_ft = base_model(pretrained=True)
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
