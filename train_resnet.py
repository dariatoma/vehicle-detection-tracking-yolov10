import os
import csv
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

data_dir = '/content/licenta/PythonProject/cropped_dataset'
num_classes = 21
batch_size = 32
num_epochs = 30
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
resnet = resnet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=learning_rate)

best_acc = 0.0

results_list = []

for epoch in range(num_epochs):
    resnet.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"\nEpoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

    resnet.eval()
    correct = 0
    total = 0
    val_running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Pentru Confusion Matrix
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_val_loss = val_running_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

    results_list.append([epoch + 1, avg_train_loss, avg_val_loss, val_accuracy])

    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(resnet.state_dict(), 'best_resnet50.pth')
        print(f"New best model saved (accuracy: {best_acc:.2f}%)")

csv_filename = 'training_results.csv'
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_Accuracy'])
    writer.writerows(results_list)

print(f"\nTraining done. Results saved to '{csv_filename}'.")
torch.save(resnet.state_dict(), 'resnet_final.pth')
print(f"Final model saved as 'resnet_final.pth'.")
print(f"Best model saved as 'best_resnet50.pth' with {best_acc:.2f}% accuracy.")

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_loader.dataset.dataset.classes)

fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(xticks_rotation=45, ax=ax, cmap='Blues')
plt.title('Confusion Matrix - Validation Set')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

