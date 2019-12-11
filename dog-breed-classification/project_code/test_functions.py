import torch
import torch.nn as nn
import myresnet as resnet
import numpy as np
from torchvision import datasets, models, transforms
from PIL import Image

datafolder = 'part'
start_fold = 0
end_fold = 1
container = np.ones((1, 2))
batch_size = 8

# Normalization for validation
data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1), ratio=(1.0,1.0)),
        transforms.ToTensor(),
    ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

network = ['ResNet-50']

def tif_to_tensor(img_path):
    with open(img_path, 'rb') as f:
        image = Image.open(f)
        image = image.convert('RGB')

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

    model_x.fc = nn.Linear(num_ftrs, 120)

    fn_best_model = modelName

    model_x.load_state_dict(torch.load(fn_best_model, map_location=lambda storage, loc: storage))
    model_x.eval()

    model_x = model_x.to(device)
    print('Trained model loaded successfully!')
    return model_x
