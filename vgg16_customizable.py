# Import libraries
import math
import time
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

import warnings

# Device configuration
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on {device} using PyTorch {torch.__version__}")

# Setting the hyper-parameters
num_hidden = 4096
num_classes = 10
num_epochs = 5
batch_size = 128
learning_rate = 1e-4
momentum = 0.9

# Define VGG architecture types, "M" represents a max pool layer
vgg_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
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

# Define VGG16 network model based on the VGG class
VGG16 = VGG(
    architecture=vgg_types["VGG16"],
    in_channels=3,
    in_height=224,
    in_width=224,
)

# Print the deep structure of convolutions, batch norms, and max pool layers of VGG16
# print(VGG16)

# Checking by passing in a dummy input represents a 3-channel 224-by-224 image
# standard_input = torch.randn((2, 3, 224, 224))
# print(VGG16(standard_input).shape)

# Loading the dataset
def load_datasets(data_dir):
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
        root=data_dir, train=True, download=True, transform=transform_train)
    
    num_train = int(np.floor(0.9 * len(dataset)))
    num_valid = len(dataset) - num_train
    train_dataset, valid_dataset = random_split(dataset, [num_train, num_valid])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)

    # load testing dataset
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

train_loader, valid_loader, test_loader = load_datasets(data_dir="./data")

# Setting model
# model = VGG16.to(device)
model = VGG16

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.005,
    momentum=momentum
)
train_losses = []
train_counter = []
valid_losses = []
valid_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(num_epochs + 1)]

# total_step = len(train_loader.dataset)

def train(model, train_loader, epoch: int):
    model.train()
    for data, target in tqdm(train_loader):
        # Move tensors to the configured device
        # data = data.to(device)
        # target = target.to(device)

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''
        print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
            epoch+1, num_epochs, i+1, total_step, loss.item()
        ))
        '''

def validate(model, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data = data.to(device)
            target = target.to(device)
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

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
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

if __name__=="__main__":
    for epoch in range(1, num_epochs + 1):
        time_start = time.time()
        train(model, train_loader, epoch)
        time_stop = time.time()
        test(model, test_loader)
        print(f"Epoch {epoch} training time: {time_stop - time_start} seconds\n")