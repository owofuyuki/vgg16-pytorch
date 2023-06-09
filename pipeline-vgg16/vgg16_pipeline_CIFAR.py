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

# Define VGG architecture types, "M" represents a max pool layer
vgg_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

# Define family of split blocks in VGG-N model
vgg_splits = {
    "SPL_VGG16_0_1": [64, 64, "M"],
    "SPL_VGG16_0_2": [64, 64, "M", 128, 128, "M"],
    "SPL_VGG16_0_3": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M"],
    "SPL_VGG16_0_4": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M"],
    "SPL_VGG16_0_5": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],  # Complete VGG16 model
    "SPL_VGG16_1_2": [128, 128, "M"],
    "SPL_VGG16_1_3": [128, 128, "M", 256, 256, 256, "M"],
    "SPL_VGG16_1_4": [128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M"],
    "SPL_VGG16_1_5": [128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "SPL_VGG16_2_3": [256, 256, 256, "M"],
    "SPL_VGG16_2_4": [256, 256, 256, "M", 512, 512, 512, "M"],
    "SPL_VGG16_2_5": [256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "SPL_VGG16_3_4": [512, 512, 512, "M"],
    "SPL_VGG16_3_5": [512, 512, 512, "M", 512, 512, 512, "M"],
    "SPL_VGG16_4_5": [512, 512, 512, "M"],
}

# Define the general VGG-N network model, "N" represents the number of layers
class VGG(nn.Module):
    def __init__(
        self,
        architecture,
        in_channels=3,
        in_height=224,
        in_width=224,
    ):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width
        self.convs = self.init_convs(architecture)
        self.fcs = self.init_fcs(architecture)

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(x.size(0), -1)
        x = self.fcs(x)
        return x

    # Implementation method for STACKING layers in a VGG-N network (the fully connected portion)
    def init_fcs(self, architecture):
        pool_count = architecture.count("M")
        factor = (2 ** pool_count)

        if (self.in_height % factor) + (self.in_width % factor) != 0:
            raise ValueError(
                f"`in_height` and `in_width` must be multiples of {factor}"
            )

        out_height = self.in_height // factor
        out_width = self.in_width // factor

        last_out_channels = next(
            x for x in architecture[::-1] if type(x) == int
        )

        return nn.Sequential(
            nn.Linear(
                last_out_channels * out_height * out_width,
                num_hidden
            ),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(num_hidden, num_classes)
        )

    # Implementation method for APPENDING layers in a VGG-N network (the convolutional portion)
    def init_convs(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers.extend(
                    [
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1)
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                    ]
                )
                in_channels = x
            else:
                layers.append(
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
                )

        return nn.Sequential(*layers)

'''
The VGG-N model used differs from the one mentioned in the original article, with some customizations.
Here, I add batch normalization to stabilize the training process and improve performance.
Also, the model above can actually handle rectangular images, not just square ones. 
Note that the in_width and in_height parameters must be a multiple of 32.
'''

# Adjust slide position
shard1_model = VGG(architecture=vgg_splits["SPL_VGG16_0_2"])
shard2_model = VGG(architecture=vgg_splits["SPL_VGG16_2_3"])
shard3_model = VGG(architecture=vgg_splits["SPL_VGG16_3_5"])

class Shard1(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Shard1, self).__init__()
        self._lock = threading.Lock()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {torch.cuda.get_device_name(self.device)}")
        self.model = shard1_model
        self.convs1 = self.model.convs.to(self.device)
        self.fcs1 = self.model.fcs.to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.convs1(x)
            # out = out.reshape(out.size(0), -1)
            # out = self.fcs1(out)
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
        self.model = shard2_model
        self.convs2 = self.model.convs.to(self.device)
        self.fcs2 = self.model.fcs.to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.convs2(x)
            # out = out.reshape(out.size(0), -1)
            # out = self.fcs2(out)
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
        self.model = shard3_model
        self.convs3 = self.model.convs.to(self.device)
        self.fcs3 = self.model.fcs.to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.convs3(x)
            out = out.reshape(out.size(0), -1)
            out = self.fcs3(out)
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