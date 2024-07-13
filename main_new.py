import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from sam import SAM
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# TODO: set seed

parser = argparse.ArgumentParser(description='SAM Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--dataset', default='cifar10')
args = parser.parse_args()

best_acc, start_epoch = 0, 0  # best test accuracy
net = ...
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net = net.to(device)

def train(optimizer, rho=0.05):
    net.train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)


if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/model:{args.model}_pretrained:{args.pretrained}_dataset:{args.dataset}_opt:{args.optimizer}_inputhessian.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    opt_state_dict = checkpoint['opt']
    sch_state_dict = checkpoint['sch']
    start_epoch = checkpoint['epoch']
for epoch in range(start_epoch, start_epoch+args.epochs):
    train_loss, train_acc = train()
    test_acc = test(epoch)
    scheduler.step()
    print("Epoch: ", epoch, "Best Acc: ", best_acc)