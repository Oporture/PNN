import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class PolyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(PolyLinear, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features, bias=bias)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.linear3 = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear1(x) + self.linear2(x) * self.linear3(x)

class SimpleMLP(nn.Module):
    def __init__(self, in_features, num_classes=1, poly=False, base_width=200):
        super(SimpleMLP, self).__init__()
        
        LinearLayer = PolyLinear if poly else nn.Linear

        self.layer1 = nn.Sequential(
            LinearLayer(in_features, base_width, bias=False),
            nn.BatchNorm1d(base_width),
            nn.ReLU(inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            LinearLayer(base_width, base_width, bias=False),
            nn.BatchNorm1d(base_width),
            nn.ReLU(inplace=True)
        )
        
        self.layer3 = nn.Sequential(
            LinearLayer(base_width, base_width, bias=False),
            nn.BatchNorm1d(base_width),
            nn.ReLU(inplace=True)
        )
        
        self.fc = nn.Linear(base_width, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Fetching Higgs dataset from OpenML (this may take a minute)....")
    try:
        higgs = fetch_openml(data_id=23512, as_frame=False, parser='auto')
        X = higgs.data
        y = np.array([int(val) for val in higgs.target])
    except Exception as e:
        print(f"Failed to fetch ID 23512 ({e}). Trying full Higgs dataset...")
        higgs = fetch_openml(name="Higgs", version=2, as_frame=False, parser='auto')
        X = higgs.data
        y = np.array([int(val) for val in higgs.target])
        
    print(f"Loaded dataset: X shape {X.shape}, y shape {y.shape}")
    
    if len(X) > 250000:
        print("Subsampling massive dataset to 250,000 samples to keep training times fast...")
        idx = np.random.choice(len(X), 250000, replace=False)
        X = X[idx]
        y = y[idx]

    in_features = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_tensor = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    test_tensor = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))

    trainloader = DataLoader(train_tensor, batch_size=256, shuffle=True, num_workers=0)
    testloader = DataLoader(test_tensor, batch_size=256, shuffle=False, num_workers=0)

    val_iterator = iter(testloader)
    
    def get_val_batch(val_iterator):
        try:
            inputs, targets = next(val_iterator)
        except StopIteration:
            val_iterator = iter(testloader)
            inputs, targets = next(val_iterator)
        return inputs, targets, val_iterator

    model1 = SimpleMLP(in_features=in_features, num_classes=1, poly=False, base_width=200).to(device)
    model2 = SimpleMLP(in_features=in_features, num_classes=1, poly=True, base_width=98).to(device)

    print(f"Model 1 (Standard MLP) Parameters: {count_parameters(model1) / 1e3:.3f} K")
    print(f"Model 2 (Polynomial MLP) Parameters: {count_parameters(model2) / 1e3:.3f} K")

    criterion = nn.BCEWithLogitsLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

    epochs = 4
    
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
            
            predicted1 = (torch.sigmoid(outputs1) >= 0.5).float()
            acc1_train = (predicted1 == targets).sum().item() / targets.size(0)
            
            optimizer2.zero_grad()
            outputs2 = model2(inputs)
            loss2 = criterion(outputs2, targets)
            loss2.backward()
            optimizer2.step()
            
            predicted2 = (torch.sigmoid(outputs2) >= 0.5).float()
            acc2_train = (predicted2 == targets).sum().item() / targets.size(0)

            model1.eval()
            model2.eval()
            with torch.no_grad():
                val_inputs, val_targets, val_iterator = get_val_batch(val_iterator)
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                
                v_outputs1 = model1(val_inputs)
                v_pred1 = (torch.sigmoid(v_outputs1) >= 0.5).float()
                acc1_val = (v_pred1 == val_targets).sum().item() / val_targets.size(0)
                
                v_outputs2 = model2(val_inputs)
                v_pred2 = (torch.sigmoid(v_outputs2) >= 0.5).float()
                acc2_val = (v_pred2 == val_targets).sum().item() / val_targets.size(0)
                
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
    
    def smooth(scalars, weight=0.99):
        if len(scalars) == 0:
            return []
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    ax1.plot(steps, smooth(model1_train_acc), label='Model 1 (Standard MLP) - Train')
    ax1.plot(steps, smooth(model2_train_acc), label='Model 2 (Polynomial MLP) - Train')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Higgs Boson Training Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(steps, smooth(model1_val_acc), label='Model 1 (Standard MLP) - Val')
    ax2.plot(steps, smooth(model2_val_acc), label='Model 2 (Polynomial MLP) - Val')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Higgs Boson Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plot_path = './higgs_training_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")

if __name__ == '__main__':
    train()
