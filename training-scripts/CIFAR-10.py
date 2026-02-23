import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class PolyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(PolyConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

    def forward(self, x):
        return self.conv1(x) + self.conv2(x) * self.conv3(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, poly=False, base_width=149):
        super(SimpleCNN, self).__init__()
        
        ConvLayer = PolyConv2d if poly else nn.Conv2d

        self.layer1 = nn.Sequential(
            ConvLayer(3, base_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            ConvLayer(base_width, base_width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.layer3 = nn.Sequential(
            ConvLayer(base_width * 2, base_width * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 4, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    def get_val_batch(val_iterator):
        try:
            inputs, targets = next(val_iterator)
        except StopIteration:
            val_iterator = iter(testloader)
            inputs, targets = next(val_iterator)
        return inputs, targets, val_iterator

    val_iterator = iter(testloader)

    model1 = SimpleCNN(poly=False, base_width=149).to(device)
    model2 = SimpleCNN(poly=True, base_width=86).to(device)

    print(f"Model 1 (Standard CNN) Parameters: {count_parameters(model1) / 1e6:.3f} M")
    print(f"Model 2 (Polynomial CNN) Parameters: {count_parameters(model2) / 1e6:.3f} M")

    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

    epochs = 20
    
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=epochs)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs)
    
    model1_train_acc = []
    model1_val_acc = []
    model2_train_acc = []
    model2_val_acc = []
    steps = []

    global_step = 0
    
    for epoch in range(epochs):
        model1.train()
        model2.train()
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer1.zero_grad()
            outputs1 = model1(inputs)
            loss1 = criterion(outputs1, targets)
            loss1.backward()
            optimizer1.step()
            
            _, predicted1 = outputs1.max(1)
            acc1_train = predicted1.eq(targets).sum().item() / targets.size(0)
            
            optimizer2.zero_grad()
            outputs2 = model2(inputs)
            loss2 = criterion(outputs2, targets)
            loss2.backward()
            optimizer2.step()
            
            _, predicted2 = outputs2.max(1)
            acc2_train = predicted2.eq(targets).sum().item() / targets.size(0)

            model1.eval()
            model2.eval()
            with torch.no_grad():
                val_inputs, val_targets, val_iterator = get_val_batch(val_iterator)
                val_inputs, val_targets, = val_inputs.to(device), val_targets.to(device)
                
                v_outputs1 = model1(val_inputs)
                _, v_pred1 = v_outputs1.max(1)
                acc1_val = v_pred1.eq(val_targets).sum().item() / val_targets.size(0)
                
                v_outputs2 = model2(val_inputs)
                _, v_pred2 = v_outputs2.max(1)
                acc2_val = v_pred2.eq(val_targets).sum().item() / val_targets.size(0)
                
            model1.train()
            model2.train()
            
            model1_train_acc.append(acc1_train * 100)
            model1_val_acc.append(acc1_val * 100)
            model2_train_acc.append(acc2_train * 100)
            model2_val_acc.append(acc2_val * 100)
            steps.append(global_step)
            global_step += 1
            
            pbar.set_postfix({
                'M1_loss': f"{loss1.item():.3f}", 'M2_loss': f"{loss2.item():.3f}", 
                'M1_tr_acc': f"{acc1_train:.3f}", 'M2_tr_acc': f"{acc2_train:.3f}"
            })

        scheduler1.step()
        scheduler2.step()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    def smooth(scalars, weight=0.90):
        if len(scalars) == 0:
            return []
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    ax1.plot(steps, smooth(model1_train_acc), label='Model 1 (Standard CNN) - Train')
    ax1.plot(steps, smooth(model2_train_acc), label='Model 2 (Polynomial CNN) - Train')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('CIFAR-10 Training Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(steps, smooth(model1_val_acc), label='Model 1 (Standard CNN) - Val')
    ax2.plot(steps, smooth(model2_val_acc), label='Model 2 (Polynomial CNN) - Val')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('CIFAR-10 Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plot_path = './training_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")

if __name__ == '__main__':
    train()
