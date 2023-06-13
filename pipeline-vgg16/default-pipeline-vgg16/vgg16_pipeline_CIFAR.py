# Import libraries
import os
import threading
import warnings

import math
import time
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef

# Device configuration
warnings.filterwarnings("ignore", category=UserWarning)

# Setting the hyper-parameters
num_hidden = 4096
num_classes = 10
num_epochs = 5
batch_size = 128
learning_rate = 1e-4
momentum = 0.9
log_interval = 10

nwt = 20
timeout = 1000000

reduction = 0.5

# Adding the arguments
parser = argparse.ArgumentParser(
    description="Pipeline Parallelism with VGG16 model based training")
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="""Total number of participating processes. 
        Should be the sum of master node and all training nodes.""")
parser.add_argument(
    "--rank",
    type=int,
    default=None,
    help="Global rank of this process. Pass in 0 for master.")
parser.add_argument(
    "--master_addr",
    type=str,
    default="localhost",
    help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""")
parser.add_argument(
    "--master_port",
    type=str,
    default="29500",
    help="""Port that master is listening on, will default to 29500 if not provided. 
        Master must be able to accept network traffic on the host and port.""")
parser.add_argument(
    "--split_size",
    type=int,
    default=8,
    help="""Split size""")
parser.add_argument(
    "--interface",
    type=str,
    default="eth0",
    help="""Interface that current device is listening on. 
        It will default to eth0 if not provided.""")

args = parser.parse_args()
assert args.rank is not None, "must provide rank argument."

class Shard1(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Shard1, self).__init__()
        self._lock = threading.Lock()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {torch.cuda.get_device_name(self.device)}")
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()).to(self.device)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)).to(self.device)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()).to(self.device)
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)).to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        return out.cpu()

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]

class Shard2(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Shard2, self).__init__()
        self._lock = threading.Lock()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {torch.cuda.get_device_name(self.device)}")
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()).to(self.device)
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()).to(self.device)
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)).to(self.device)
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()).to(self.device)
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()).to(self.device)
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)).to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.layer5(x)
            out = self.layer6(out)
            out = self.layer7(out)
            out = self.layer8(out)
            out = self.layer9(out)
            out = self.layer10(out)
        return out.cpu()

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]

class Shard3(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Shard3, self).__init__()
        self._lock = threading.Lock()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {torch.cuda.get_device_name(self.device)}")
        
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()).to(self.device)
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()).to(self.device)
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)).to(self.device)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU()).to(self.device)
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()).to(self.device)
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes)).to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.layer11(x)
            out = self.layer12(out)
            out = self.layer13(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            out = self.fc1(out)
            out = self.fc2(out)
        return out.cpu()

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]

class DistNet(nn.Module):
    """
    Assemble all parts as an nn.Module and define pipelining logic
    """
    def __init__(self, split_size, workers, *args, **kwargs):
        super(DistNet, self).__init__()

        self.split_size = split_size

        self.p1_rref = rpc.remote(
            workers[0],
            Shard1,
            args = args,
            kwargs = kwargs,
            timeout = 0
        )
        self.p2_rref = rpc.remote(
            workers[1],
            Shard2,
            args = args,
            kwargs = kwargs,
            timeout = 0
        )
        self.p3_rref = rpc.remote(
            workers[2],
            Shard3,
            args = args,
            kwargs = kwargs,
            timeout = 0
        )

    def forward(self, xs):
        out_futures = []
        for x in iter(xs.split(self.split_size, dim=0)):
            x1_rref = RRef(x)
            x2_rref = self.p1_rref.remote().forward(x1_rref)
            x3_rref = self.p2_rref.remote().forward(x2_rref)
            x4_fut = self.p3_rref.rpc_async().forward(x3_rref)
            out_futures.append(x4_fut)

        # Collect and cat all output tensors into one tensor
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p3_rref.remote().parameter_rrefs().to_here())
        return remote_params
    
# Run RPC Processes
def run_master(split_size):
    # define transforms
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # load training dataset and validation dataset
    dataset = datasets.CIFAR10(
        root="../data", train=True, download=True, transform=transform_train)
    
    num_train = int(np.floor(0.9 * len(dataset)))
    num_valid = len(dataset) - num_train
    train_dataset, valid_dataset = random_split(dataset, [num_train, num_valid])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # load testing dataset
    test_dataset = datasets.CIFAR10(
        root="../data", train=False, download=True, transform=transform_test)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # put the two model parts on worker1 and worker2 respectively
    model = DistNet(split_size, ["worker0", "worker1", "worker2"])
    print('Split size =',split_size)
    criterion = nn.CrossEntropyLoss()
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )
    train_losses = []
    train_counter = []
    valid_losses = []
    valid_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(num_epochs + 1)]
    
    # total_step = len(train_loader.dataset)

    def train(epoch):
        model.train()
        for data, target in tqdm(train_loader):
            with dist_autograd.context() as context_id:
                output = model(data)
                loss = [criterion(output, target)]
                dist_autograd.backward(context_id, loss)
                opt.step(context_id)

    def validate():
        model.eval()
        valid_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in valid_loader:
                output = model(data)
                valid_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()

        valid_loss /= len(valid_loader.dataset)
        valid_losses.append(valid_loss)
        print("Validation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            valid_loss, correct, len(valid_loader.dataset),
            100. * correct / len(valid_loader.dataset)
        ))
    
    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print("Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
        ))

    for epoch in range(1, num_epochs + 1):
        time_start = time.time()
        train(epoch)
        time_stop = time.time()
        test()
        print(f"Epoch {epoch} training time: {time_stop - time_start} seconds\n")

def run_worker(rank, world_size, num_split):
    # higher timeout is added to accommodate for kernel compilation time in case of ROCm.
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=nwt, rpc_timeout=timeout)

    if rank == 0:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master(num_split)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()

if __name__=="__main__":
    # os.system(f"tegrastats --interval 1000 --logfile ./log/split_{args.split_size}/master_usage.log &")
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['GLOO_SOCKET_IFNAME'] = args.interface
    os.environ["TP_SOCKET_IFNAME"] = args.interface

    run_worker(rank=args.rank, world_size=args.world_size, num_split=args.split_size)

    # os.system(f"pkill -f \"tegrastats\"")