import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class PolyConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(PolyConv1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

    def forward(self, x):
        return self.conv1(x) + self.conv2(x) * self.conv3(x)

class SimpleTextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes=1, poly=False, base_width=128):
        super(SimpleTextCNN, self).__init__()
        self.embed_dim = 128
        self.embedding = nn.Embedding(vocab_size, self.embed_dim)
        
        ConvLayer = PolyConv1d if poly else nn.Conv1d

        self.layer1 = nn.Sequential(
            ConvLayer(self.embed_dim, base_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        
        self.layer2 = nn.Sequential(
            ConvLayer(base_width, base_width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        
        self.layer3 = nn.Sequential(
            ConvLayer(base_width * 2, base_width * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_width * 4, num_classes)

    def forward(self, x):
        x = self.embedding(x) 
        x = x.transpose(1, 2) 
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def count_encoder_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embed = sum(p.numel() for p in model.embedding.parameters() if p.requires_grad)
    return total - embed

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading BoolQ dataset (this may take a minute)...")
    dataset = load_dataset("boolq")
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    def tokenize_function(examples):
        return tokenizer(examples['question'], examples['passage'], 
                         truncation=True, padding='max_length', max_length=256)
    
    print("Tokenizing data...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]
    
    def collate_fn(batch):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        labels = torch.tensor([float(item['answer']) for item in batch]).unsqueeze(1)
        return input_ids, labels

    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0)
    testloader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    val_iterator = iter(testloader)
    
    def get_val_batch(val_iterator_):
        try:
            inputs, targets = next(val_iterator_)
        except StopIteration:
            val_iterator_ = iter(testloader)
            inputs, targets = next(val_iterator_)
        return inputs, targets, val_iterator_

    vocab_size = tokenizer.vocab_size

    model1 = SimpleTextCNN(vocab_size=vocab_size, num_classes=1, poly=False, base_width=256).to(device)
    model2 = SimpleTextCNN(vocab_size=vocab_size, num_classes=1, poly=True, base_width=148).to(device)

    print(f"Model 1 (Standard 1D CNN) Encoder Params: {count_encoder_parameters(model1) / 1e3:.3f} K")
    print(f"Model 2 (Polynomial 1D CNN) Encoder Params: {count_encoder_parameters(model2) / 1e3:.3f} K")

    criterion = nn.BCEWithLogitsLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=0.0001)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.0001)

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

    ax1.plot(steps, smooth(model1_train_acc), label='Model 1 (Standard 1D CNN) - Train')
    ax1.plot(steps, smooth(model2_train_acc), label='Model 2 (Polynomial 1D CNN) - Train')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('BoolQ Training Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(steps, smooth(model1_val_acc), label='Model 1 (Standard 1D CNN) - Val')
    ax2.plot(steps, smooth(model2_val_acc), label='Model 2 (Polynomial 1D CNN) - Val')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('BoolQ Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plot_path = './boolq_training_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")

if __name__ == '__main__':
    train()
