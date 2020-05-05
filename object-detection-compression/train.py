import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets

from cifar10_models import *

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch-size', '-N', type=int, default=200, help='batch size')
parser.add_argument(
    '--train', '-f', required=True, type=str, help='folder of training images')
parser.add_argument(
    '--max-epochs', '-e', type=int, default=200, help='max epochs')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
parser.add_argument('--disable-cuda', action='store_true',
    help='Disable CUDA')
parser.add_argument(
    '--iterations', type=int, default=1, help='unroll iterations')
parser.add_argument('--checkpoint', type=int, help='unroll iterations')
args = parser.parse_args()

args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

## load 32x32 patches from images

train_transform = transforms.Compose([
    transforms.CenterCrop((32, 32)),
    transforms.ToTensor()
])

normalize =  transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])

# train_set = dataset.ImageFolder(root=args.train, transform=train_transform)
train_set = datasets.CIFAR10(args.train, train=True, transform=train_transform, download=True)
train_loader = data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                               drop_last=True, pin_memory=True, num_workers=16)

print('total images: {}; total batches: {}'.format(
    len(train_set), len(train_loader)))

## load networks on GPU
import network

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

output_disc = freeze_network(inception_v3(pretrained=True)).to(args.device)

encoder = network.EncoderCell().to(args.device)
binarizer = network.Binarizer().to(args.device)
decoder = network.DecoderCell().to(args.device)

solver = optim.Adam(
    [
        {
            'params': encoder.parameters()
        },
        {
            'params': binarizer.parameters()
        },
        {
            'params': decoder.parameters()
        },
    ],
    lr=args.lr)


def resume(epoch=None):
    if epoch is None:
        s = 'iter'
        epoch = 0
    else:
        s = 'epoch'

    encoder.load_state_dict(
        torch.load('checkpoint/encoder_{}_{:08d}.pth'.format(s, epoch)))
    binarizer.load_state_dict(
        torch.load('checkpoint/binarizer_{}_{:08d}.pth'.format(s, epoch)))
    decoder.load_state_dict(
        torch.load('checkpoint/decoder_{}_{:08d}.pth'.format(s, epoch)))


def save(index, epoch=True):
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')

    if epoch:
        s = 'epoch'
    else:
        s = 'iter'

    torch.save(encoder.state_dict(), 'checkpoint/encoder_{}_{:08d}.pth'.format(
        s, index))

    torch.save(binarizer.state_dict(),
               'checkpoint/binarizer_{}_{:08d}.pth'.format(s, index))

    torch.save(decoder.state_dict(), 'checkpoint/decoder_{}_{:08d}.pth'.format(
        s, index))


# resume()
if __name__ == '__main__':
    scheduler = LS.MultiStepLR(solver, milestones=[3, 10, 20, 50, 100], gamma=0.5)

    last_epoch = 0
    if args.checkpoint:
        resume(args.checkpoint)
        last_epoch = args.checkpoint
        scheduler.last_epoch = last_epoch - 1
    celoss = nn.CrossEntropyLoss().to(args.device)
    l2loss = nn.MSELoss().to(args.device)
    for epoch in range(last_epoch + 1, args.max_epochs + 1):

        scheduler.step()

        for batch, data in enumerate(train_loader):
            batch_t0 = time.time()
            labels = data[1].to(args.device)
            data = data[0]

            ## init lstm state
            encoder_h_1 = (Variable(torch.zeros(data.size(0), 256, 8, 8).to(args.device)),
                           Variable(torch.zeros(data.size(0), 256, 8, 8).to(args.device)))
            encoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).to(args.device)),
                           Variable(torch.zeros(data.size(0), 512, 4, 4).to(args.device)))
            encoder_h_3 = (Variable(torch.zeros(data.size(0), 512, 2, 2).to(args.device)),
                           Variable(torch.zeros(data.size(0), 512, 2, 2).to(args.device)))

            decoder_h_1 = (Variable(torch.zeros(data.size(0), 512, 2, 2).to(args.device)),
                           Variable(torch.zeros(data.size(0), 512, 2, 2).to(args.device)))
            decoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).to(args.device)),
                           Variable(torch.zeros(data.size(0), 512, 4, 4).to(args.device)))
            decoder_h_3 = (Variable(torch.zeros(data.size(0), 256, 8, 8).to(args.device)),
                           Variable(torch.zeros(data.size(0), 256, 8, 8).to(args.device)))
            decoder_h_4 = (Variable(torch.zeros(data.size(0), 128, 16, 16).to(args.device)),
                           Variable(torch.zeros(data.size(0), 128, 16, 16).to(args.device)))

            patches = Variable(data.to(args.device))

            solver.zero_grad()

            losses = []

            res = patches - 0.5

            bp_t0 = time.time()

            for _ in range(args.iterations):
                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                    res, encoder_h_1, encoder_h_2, encoder_h_3)

                codes = binarizer(encoded)

                output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                    codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)


                res = celoss(output_disc(norm(output + 0.5, [0.4914, 0.4822, 0.4465],
                             [0.229, 0.224, 0.225])), labels)
                # + l2loss(output, data.to(args.device)).mean()
                losses.append(res.mean())

            bp_t1 = time.time()

            loss = sum(losses) / args.iterations
            loss.backward()

            solver.step()

            batch_t1 = time.time()

            print(
                '[TRAIN] Epoch[{}]({}/{}); Loss: {:.6f}; Backpropagation: {:.4f} sec; Batch: {:.4f} sec'.
                format(epoch, batch + 1,
                       len(train_loader), loss.item(), bp_t1 - bp_t0, batch_t1 -
                       batch_t0))
            print(('{:.4f} ' * args.iterations +
                   '\n').format(* [l.item() for l in losses]))

            index = (epoch - 1) * len(train_loader) + batch

            ## save checkpoint every 500 training steps
            #if index % 500 == 0:
            #    save(0, False)

        save(epoch)

