'''Train CIFAR10 with PyTorch.''' # Source: https://github.com/kuangliu/pytorch-cifar/tree/master
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from functorch import vmap, vjp, jvp, jacrev
from sam import SAM
from ssam import SSAM
import os
import random
import argparse
from copy import deepcopy
from models import resnet, cnn
from utils import *
import gc

# Training
def train(epoch, rho=0.05, ssam_lambda=0.5):
    optimizer.set_rho(rho)
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss, correct, total = 0, 0, 0
    if args.optimizer == 'ssam':
        ssam_sharp, sam_sharp = 0., 0.
        ascent_descent_cosine_sim, ssam_sam_ascent_cosine_sim = 0., 0.
        descent_direction_sparsity = 0.
        ssam_asc_grad_coherence, sam_asc_grad_coherence, ssam_desc_grad_coherence = 0., 0., 0.
        ssam_sam_ascent_l1_diff = 0.
        # grad_hessian_alignment = 0.
        lambda_1 = 0.
        optimizer.set_lambda(ssam_lambda)
    elif args.optimizer == 'sam': # TODO: Write metrics to extract for SAM
        pass

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # SSAM
        if args.optimizer == 'ssam':
            inputs_, targets_, inputs_1, targets_1, inputs_2, targets_2, inp3, targ3 = deepcopy(inputs), deepcopy(targets), deepcopy(inputs), deepcopy(targets), deepcopy(inputs), deepcopy(targets), deepcopy(inputs), deepcopy(targets)
            enable_bn(net) # Preparation
            loss = criterion(outputs, targets)
            inputs_prep, inputs_copy, targets_copy = deepcopy(inputs), deepcopy(inputs), deepcopy(targets)
            copy_of_net = deepcopy(net)
            copy_of_optimizer = SAM(copy_of_net.parameters(), optim.SGD, rho=0.05, lr=args.lr, momentum=0.9, weight_decay=5e-4)
            outputs_1 = copy_of_net(inputs_1)
            outputs = net(inputs_copy)       

            loss_1 = criterion(outputs_1, targets_1)
            loss_1.backward()
            sam_ascent_dir = {}
            for name, param in copy_of_net.named_parameters():
                if param.grad is not None:
                    sam_ascent_dir[name] = torch.flatten(param.grad).detach()
            _ = copy_of_optimizer.first_step(zero_grad=True)
            sam_loss = criterion(copy_of_net(inputs_2), targets_2).item() 

            l = ell(outputs, targets_copy)
            l2 = ell_2(outputs)

            nabla_f = compute_jacobian(net.module.to("cuda:0"), inp3.to("cuda:0"))
            _, ascent_dirs = optimizer.first_step(zero_grad=True, n_iter=5, ell2=l2, nabla_f=nabla_f, parameters=net.named_parameters())
              # Compute metrics
            ascent_descent_cosine_sim += dict_cosine_similarity(ascent_dirs, descent_dirs)
            ssam_sam_ascent_cosine_sim += dict_cosine_similarity(ascent_dirs, sam_ascent_dir)
            ssam_sam_ascent_l1_diff += dict_l1_difference(ascent_dirs, sam_ascent_dir)
            descent_direction_sparsity += l1_dist_to_uniform(descent_dirs)
            ssam_asc_grad_coherence += 0 if batch_idx == 0 else dict_cosine_similarity(prev_ascent_dirs, ascent_dirs)
            sam_asc_grad_coherence += 0 if batch_idx == 0 else dict_cosine_similarity(sam_ascent_dir, prev_sam_ascent_dirs)
            ssam_desc_grad_coherence += 0 if batch_idx == 0 else dict_cosine_similarity(ascent_dirs, prev_descent_dirs)
            # top_eigenvalue, _, _ = get_hessian_info(deepcopy(net), criterion, (inp3, targ3), True)
            # lambda_1 += 0#np.mean(top_eigenvalue)
            prev_descent_dirs = descent_dirs
            prev_ascent_dirs = ascent_dirs
            prev_sam_ascent_dirs = sam_ascent_dir
        # SAM
        elif args.optimizer == 'sam':
            inputs_, targets_, inputs_1, targets_1, inputs_2, targets_2, inp3, targ3 = deepcopy(inputs), deepcopy(targets), deepcopy(inputs), deepcopy(targets), deepcopy(inputs), deepcopy(targets), deepcopy(inputs), deepcopy(targets)
            enable_bn(net) # Preparation
            loss = criterion(outputs, targets)
            loss.backward()
            ascent_dirs = optimizer.first_step(zero_grad=True)
        # SGD
        elif args.optimizer == 'sgd':
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        outputs = net(inputs)
        if args.optimizer == 'sam' or args.optimizer == 'ssam': # SAM or SSAM
            disable_bn(net)
            ssam_loss = criterion(net(inputs_), targets_)
            criterion(net(inputs_), targets_).backward()
            descent_dirs = {}
            for name, param in net.named_parameters():
                if param.grad is not None:
                    descent_dirs[name] = torch.flatten(param.grad).detach()
            # net.train()
            optimizer.second_step(zero_grad=True)
            if args.optimizer == 'ssam':
                ssam_sharp += ssam_loss.item() 
                sam_sharp += sam_loss
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if args.optimizer != 'ssam': 
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        else: 
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Avg Incr. Sharp %.1f | Lambda 1 %.1f | SSAM/SAM Ascent Sim %.1f | SSAM/SAM L1 %.1f | Descent Dir Sparsity %.1f | Ascent Descent Cosine Sim %.1f | SSAM Asc Grad Cohere %.1f | SSAM Desc Grad Cohere %.1f | SAM Asc Grad Cohere %.1f'% 
                           (train_loss/(batch_idx+1), 100.*correct/total, correct, total, (ssam_sharp - sam_sharp)/(batch_idx+1), lambda_1/(batch_idx+1), ssam_sam_ascent_cosine_sim/(batch_idx+1), ssam_sam_ascent_l1_diff/(batch_idx+1), descent_direction_sparsity/(batch_idx+1),
                            ascent_descent_cosine_sim/(batch_idx+1), ssam_asc_grad_coherence/(batch_idx+1), ssam_desc_grad_coherence/(batch_idx+1), sam_asc_grad_coherence/(batch_idx+1)))
            with open(f"logs/model:{args.model}_pretrained:{args.pretrained}_dataset:{args.dataset}_opt:{args.optimizer}.txt", "a") as f:
                if batch_idx == 0:
                    if args.optimizer == 'ssam':
                        f.write("epoch,batch_idx,train_loss,ssam_sharp,sam_sharp,rho,ascent_descent_cosine_sim,ssam_sam_ascent_cosine_sim,ssam_sam_ascent_l1_diff,descent_direction_sparsity,ssam_asc_grad_coherence,sam_asc_grad_coherence,ssam_desc_grad_coherence,lambda_1\n")
                    else: 
                        f.write("epoch,batch_idx,train_loss,rho\n")
                if args.optimizer == 'ssam': 
                    f.write(f"{epoch}, {batch_idx},{train_loss/(batch_idx+1)},{ssam_sharp/(batch_idx+1)},{sam_sharp/(batch_idx+1)},{rho},{ascent_descent_cosine_sim/(batch_idx+1)},{ssam_sam_ascent_cosine_sim/(batch_idx+1)},{ssam_sam_ascent_l1_diff/(batch_idx+1)},{descent_direction_sparsity/(batch_idx+1)},{ssam_asc_grad_coherence/(batch_idx+1)},{sam_asc_grad_coherence/(batch_idx+1)},{ssam_desc_grad_coherence/(batch_idx+1)},{lambda_1/(batch_idx+1)}\n")
                else: f.write(f"{batch_idx},{train_loss},{rho}\n")
        
def test(epoch):
    global best_acc
    net.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {'net': net.state_dict(), 'acc': acc, 'epoch': epoch, 'opt': optimizer.state_dict(), 'sch': scheduler.state_dict()}
        if not os.path.isdir('checkpoint'): os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/model:{args.model}_pretrained:{args.pretrained}_dataset:{args.dataset}_opt:{args.optimizer}.pth')
        best_acc = acc
    return acc

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--ssam_lambda', default=0.5)
parser.add_argument('--model', default='resnet18')
parser.add_argument('--pretrained', default=True)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--dataset', default='cifar10')
args = parser.parse_args()

print('==> Preparing data..')
if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=4) #batch_size originally 128
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
elif args.dataset == 'cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    
    classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
        'mushrooms', 'oak_tree', 'oranges', 'orchid', 'otter', 'palm_tree', 'pear', 
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 
        'worm')
elif args.dataset == 'mnist':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    classes = tuple(str(i) for i in range(10))
elif args.dataset == 'svhn':
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    classes = tuple(str(i) for i in range(10))

criterion = nn.CrossEntropyLoss()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, start_epoch = 0, 0  # best test accuracy
net = cnn.CNN()
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net = net.to(device)
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/model:{args.model}_pretrained:{args.pretrained}_dataset:{args.dataset}_opt:{args.optimizer}_inputhessian.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    opt_state_dict = checkpoint['opt']
    sch_state_dict = checkpoint['sch']
    start_epoch = checkpoint['epoch']
if args.optimizer == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.resume: optimizer.load_state_dict(opt_state_dict)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, last_epoch=start_epoch-1)    
    if args.resume: scheduler.load_state_dict(sch_state_dict)
elif args.optimizer == 'sam':
    optimizer = SAM(net.parameters(), optim.SGD, rho=0.05, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.resume: optimizer.load_state_dict(opt_state_dict)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=200, last_epoch=start_epoch-1)  
    if args.resume: scheduler.load_state_dict(sch_state_dict)
else:
    optimizer = SSAM(net.parameters(), optim.SGD, rho=0.1, lr=args.lr, momentum=0.9, weight_decay=5e-4, lam=args.ssam_lambda)
    if args.resume: optimizer.load_state_dict(opt_state_dict)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=200, last_epoch=start_epoch-1)    
    if args.resume: scheduler.load_state_dict(sch_state_dict)

for epoch in range(start_epoch, start_epoch+args.epochs):
    if args.optimizer == 'ssam': 
        rho=0.05
        train_loss, train_acc, ssam_sharp, sam_sharp, ascent_descent_cosine_sim, ssam_sam_ascent_cosine_sim, descent_direction_sparsity, ssam_asc_grad_coherence, sam_asc_grad_coherence, ssam_desc_grad_coherence, lambda_1, ssam_sam_ascent_l1_diff = train(epoch, ssam_lambda=0.5, rho=rho)
    else: 
        rho=0.05
        train_loss, train_acc = train(epoch, rho=rho)
    test_acc = test(epoch)
    scheduler.step()
    print("Epoch: ", epoch, "Best Acc: ", best_acc)
