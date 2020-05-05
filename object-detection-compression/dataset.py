# modified from https://github.com/desimone/vision/blob/fb74c76d09bcc2594159613d5bdadd7d4697bb11/torchvision/datasets/folder.py

import os
import os.path

import torch
from torchvision import transforms, models
import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


def norm(tensor, mean, std):
    tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, root, transform=None, loader=default_loader, device=torch.device('cuda')):
        images = []
        label = 0
        labels = []
        for filename in os.listdir(root):
            dirr = os.path.join(root, filename)
            if os.path.isdir(dirr):
                for f in os.listdir(dirr):
                    if is_image_file(f):
                        images.append('{}'.format(os.path.join(dirr, f)))
                        labels.append(-1)
                label += 1

        self.root = root
        self.imgs = images
        self.labels = labels
        self.transform = transform
        self.loader = loader
        self.vgg = (models.vgg16(pretrained=True))
        self.device = device

    def __getitem__(self, index):
        filename = self.imgs[index]
        label = self.labels[index]
        try:
            img = self.loader(filename)
        except:
            return torch.zeros((3, 32, 32))

        if self.transform is not None:
            img = self.transform(img)
            img2 = norm(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)

        if label == -1:
            prediction = self.vgg(img2)  # Returns a Tensor of shape (batch, num class labels)
            self.labels[index] = prediction.data.numpy().argmax()  # Our prediction will be the index of the class label with the largest value.

        return img, torch.tensor(self.labels[index], dtype=int)

    def __len__(self):
        return len(self.imgs)
