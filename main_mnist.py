from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import numpy as np
import cv2
from tqdm import tqdm
from torchinfo import summary
import albumentations as A
from albumentations.pytorch import ToTensorV2


###################
# Neural Network Architecture
###################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # First convolutional block: 28x28 -> 14x14.
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Second convolution: 14x14 -> 14x14.
        self.conv2 = nn.Sequential(
            nn.Conv2d(2, 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )

        # Third convolutional block: 14x14 -> 7x7.
        self.conv3 = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Fourth convolutional block: 7x7 -> 3x3.
        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Fifth convolution: 3x3 -> 3x3.
        self.conv5 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # 3x3x16 -> 144 -> 10 logits.
        self.fc1 = nn.Linear(16 * 3 * 3, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.reshape(-1, 16 * 3 * 3)
        x = self.fc1(x)
        return x


###################
# Device Configuration
###################

use_cuda = torch.cuda.is_available()
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if use_cuda else "cpu")


###################
# Data Augmentation
###################

# Training augmentation pipeline
train_transforms = A.Compose([
    A.ShiftScaleRotate(
        shift_limit=0.0625,
        scale_limit=0.05,
        rotate_limit=12,
        p=0.5,
        border_mode=cv2.BORDER_CONSTANT,
        value=0
    ),
    A.Normalize(
        mean=[0.1307],
        std=[0.3081],
    ),
    ToTensorV2(),
])

# Test transforms (only normalization)
test_transforms = A.Compose([
    A.Normalize(
        mean=[0.1307],
        std=[0.3081],
    ),
    ToTensorV2(),
])


###################
# Custom Dataset
###################

class MNISTAlbumentations(datasets.MNIST):
    def __init__(self, root, train=True, download=True, transform=None):
        super().__init__(root, train=train, download=download, transform=None)
        self.transform = transform

    @property
    def raw_folder(self):
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "MNIST", "processed")
        
    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]
        img = np.array(img)
        img = np.expand_dims(img, axis=-1)  # Add channel dimension for Albumentations
        
        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]
            
        return img, label


###################
# Utility Functions
###################

def visualize_augmentations(dataset, idx=0, samples=5):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(20, 4))
    for i in range(samples):
        data = dataset[idx][0]
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        if data.shape[0] == 1:  # If channels first, move to last
            data = np.transpose(data, (1, 2, 0))
        plt.subplot(1, samples, i + 1)
        plt.imshow(data.squeeze(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


###################
# Training Functions
###################

def train(model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()
    pbar = tqdm(train_loader)
    total = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()
            
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        total += len(target)
        correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / total
        
        pbar.set_description(desc= f'Training set: loss={loss.item():.5f} accuracy={accuracy:.2f}% batch_id={batch_idx}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.5f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


###################
# Main Execution
###################

if __name__ == '__main__':
    # Initial setup
    use_cuda = torch.cuda.is_available()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if use_cuda else "cpu")
    torch.manual_seed(14596)
    
    # Training parameters
    batch_size = 512
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type in ["cuda", "mps"] else {}
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        MNISTAlbumentations('../data', train=True, download=True, transform=train_transforms),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        MNISTAlbumentations('../data', train=False, transform=test_transforms),
        batch_size=batch_size, shuffle=True, **kwargs)

    # Initialize model and print summary
    model = Net().to(device)
    summary(model, input_size=(1, 1, 28, 28), device=device)
    
    # Training configuration
    results = []
    num_epochs = 20
    max_lr = 0.4 
    initial_div = 25 
    final_div = 175 
    warmup_pct = 0.5 
    momentum = 0.95 
    weight_decay = 0.0005 

    print(f"\nTraining with parameters:")
    print(f"max_lr: {max_lr}, initial_div: {initial_div}, final_div: {final_div}")
    print(f"warmup_pct: {warmup_pct}, momentum: {momentum}, weight_decay: {weight_decay}")

    # Initialize model and optimizer
    model = Net().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=max_lr/initial_div,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # Setup OneCycle scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=warmup_pct,
        div_factor=initial_div,
        final_div_factor=final_div
    )

    # Training Loop
    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'---- Epoch : {epoch:02d} |  learning_rate : {current_lr:.7f} ----')
        
        train(model, device, train_loader, optimizer, epoch, scheduler)
        test(model, device, test_loader)
