import argparse
import os
import numpy as np
from imageio import imread, imsave

import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import CIFAR10

from cifar10_models import *

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', '-m', required=True, type=str, help='path to model')
parser.add_argument(
    '--input', '-i', required=True, type=str, help='input image')
parser.add_argument(
    '--output', '-o', required=True, type=str, help='output codes')
parser.add_argument(
    '--true', '-t', required=True, type=str, help='output codes')
parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument(
    '--iterations', type=int, default=1, help='unroll iterations')
args = parser.parse_args()

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    return img[starty:starty+cropy, startx:startx+cropx, :]

train_transform = transforms.Compose([
    transforms.CenterCrop((32, 32)),
    transforms.ToTensor()
])


# train_set = dataset.ImageFolder(root=args.train, transform=train_transform)
train_set = CIFAR10(args.input, train=False, transform=train_transform, download=True)
train_loader = data.DataLoader(dataset=train_set, batch_size=200, shuffle=False,
                               drop_last=True, pin_memory=False, num_workers=16)

def freeze_network(net):
    for child in net.children():
        for param in child.parameters():
            param.requires_grad = False
    return net

def norm(tensor, mean, std):
    tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

tags = unpickle('batches.meta')[b'label_names']

# image = imread(args.input, pilmode='RGB')
# image = crop_center(image, 32, 32)
# print(image.shape)
# image = torch.from_numpy(
#     np.expand_dims(
#         np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)), 0))
# batch_size, input_channels, height, width = image.size()
# assert height % 32 == 0 and width % 32 == 0

height, width = (32, 32)
output_disc = freeze_network(inception_v3(pretrained=True))
idx = 0

t_a = 0
a = 0
a_ta = 0


import network

encoder = network.EncoderCell()
binarizer = network.Binarizer()
decoder = network.DecoderCell()

encoder.eval()
binarizer.eval()
decoder.eval()

encoder.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
binarizer.load_state_dict(
    torch.load(args.model.replace('encoder', 'binarizer'), map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load(args.model.replace('encoder', 'decoder'), map_location=torch.device('cpu')))

marks = [0 for _ in range(10)]

for image, label in train_loader:

    image = Variable(image.cpu())

    encoder_h_1 = (Variable(
        torch.zeros(image.size(0), 256, height // 4, width // 4), volatile=True),
                   Variable(
                       torch.zeros(image.size(0), 256, height // 4, width // 4),
                       volatile=True))
    encoder_h_2 = (Variable(
        torch.zeros(image.size(0), 512, height // 8, width // 8), volatile=True),
                   Variable(
                       torch.zeros(image.size(0), 512, height // 8, width // 8),
                       volatile=True))
    encoder_h_3 = (Variable(
        torch.zeros(image.size(0), 512, height // 16, width // 16), volatile=True),
                   Variable(
                       torch.zeros(image.size(0), 512, height // 16, width // 16),
                       volatile=True))

    decoder_h_1 = (Variable(
        torch.zeros(image.size(0), 512, height // 16, width // 16), volatile=True),
                   Variable(
                       torch.zeros(image.size(0), 512, height // 16, width // 16),
                       volatile=True))
    decoder_h_2 = (Variable(
        torch.zeros(image.size(0), 512, height // 8, width // 8), volatile=True),
                   Variable(
                       torch.zeros(image.size(0), 512, height // 8, width // 8),
                       volatile=True))
    decoder_h_3 = (Variable(
        torch.zeros(image.size(0), 256, height // 4, width // 4), volatile=True),
                   Variable(
                       torch.zeros(image.size(0), 256, height // 4, width // 4),
                       volatile=True))
    decoder_h_4 = (Variable(
        torch.zeros(image.size(0), 128, height // 2, width // 2), volatile=True),
                   Variable(
                       torch.zeros(image.size(0), 128, height // 2, width // 2),
                       volatile=True))

    # if args.cuda:
    #     encoder = encoder.cuda()
    #     binarizer = binarizer.cuda()
    #     decoder = decoder.cuda()
    #
    #     image = image.cuda()
    #
    #     encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
    #     encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
    #     encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())
    #
    #     decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
    #     decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
    #     decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
    #     decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())
    codes = []
    res = image - 0.5
    for iters in range(args.iterations):
        encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
            res, encoder_h_1, encoder_h_2, encoder_h_3)

        code = binarizer(encoded)

        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
            code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

        res = res - output
        codes.append(code.data.cpu().numpy())

        print('Iter: {:02d}; Loss: {:.06f}'.format(iters, res.data.abs().mean()))

    # codes = (np.stack(codes).astype(np.int8) + 1) // 2
    #
    # export = np.packbits(codes.reshape(-1))

    pred = np.argmax(output_disc(norm(output + 0.5, [0.4914, 0.4822, 0.4465],
                             [0.229, 0.224, 0.225])).data.cpu().numpy(), 1)

    image_pred = np.argmax(output_disc(norm(image, [0.4914, 0.4822, 0.4465],
                             [0.229, 0.224, 0.225])).data.cpu().numpy(), 1)

    for i in range(image.size(0)):
        code_temp = (codes[0][i, :, :, :].astype(np.int8) + 1) // 2
        export = np.packbits(code_temp.reshape(-1))
        temp = output[i, :, :, :]
        image_temp = image[i, :, :, :]
        np.savez_compressed(os.path.join(args.output, str(idx)), shape=code_temp.shape, codes=export)


        if pred[i] == label[i]:
            a += 1
            if marks[pred[i]] == 0:
                marks[pred[i]] = 1
                imsave(os.path.join(args.output, '{}.png'.format(tags[pred[i]])),
                   np.squeeze((temp + 0.5).data.cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
                   .transpose(1, 2, 0))

        if image_pred[i] == label[i]:
            t_a += 1

        if pred[i] == image_pred[i]:
            a_ta += 1

        imsave(os.path.join(args.true, '{}_{:02d}.png'.format(idx, iters)),
        np.squeeze(image_temp.numpy().clip(0, 1) * 255.0).astype(np.uint8)
        .transpose(1, 2, 0))
        idx += 1
    print(idx)
print('Statistics')
print('Accuracy after compression')
print(a/len(train_loader) * 100)
print('Accuracy of inception')
print(t_a/len(train_loader) * 100)
print('Accuracy of Compression and Inception')
print(a_ta/len(train_loader) * 100)