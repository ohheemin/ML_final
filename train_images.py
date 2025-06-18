import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt 

#================hyper parameter=============================

BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.001

TRAIN_DIR = "/home/ohheemin/dataset2/train"
TEST_DIR = "/home/ohheemin/dataset2/test"

#==================setting GPU=============================== 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#================data augmentation============================

train_transform = transforms.Compose([
    transforms.Resize((72, 72)),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

#==================loading the data============================

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError("train folder does not exist")
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError("test folder does not exist")
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#====================class name===============================

class_names = train_dataset.classes
num_classes = len(class_names)

#================Conv-BN-SiLU Block===========================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x)

#===================C2f Block==================================

class C2fBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super(C2fBlock, self).__init__()
        hidden_channels = out_channels // 2
        self.stem = ConvBlock(in_channels, hidden_channels, kernel_size=1, padding=0)
        self.blocks = nn.Sequential(*[
            ConvBlock(hidden_channels, hidden_channels) for _ in range(num_blocks)
        ])
        self.concat = ConvBlock(in_channels + hidden_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        y1 = self.stem(x)
        y2 = self.blocks(y1)
        out = torch.cat([x, y2], dim=1)
        return self.concat(out)

#===========YOLOv8 style, classifier==========================

class YOLOv8Classifier(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv8Classifier, self).__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3),                   
            C2fBlock(32, 64, num_blocks=1),                    
            nn.MaxPool2d(2),                                 
            C2fBlock(64, 128, num_blocks=2),                
            nn.MaxPool2d(2),                                   
            C2fBlock(128, 256, num_blocks=2),                 
            nn.MaxPool2d(2),                                  
            ConvBlock(256, 512, kernel_size=1, padding=0),    
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)  
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

#==============making learning model==========================

model = YOLOv8Classifier(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#==================training loop=============================

train_losses = []
train_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(accuracy)

    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f}, Train Accuracy: {accuracy:.2f}%")

#==============test accuracy evaluation=======================

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f"Final Test Accuracy: {test_accuracy:.2f}%")

#===================saving the model============================

torch.save(model.state_dict(), "0616_augmented.pth")

#======================plotting==============================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), train_accuracies, label='Train Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig("0612_training_curves.png")  
plt.show()
