import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models.ensemble import CombinedModel

# --- CONFIG ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "best.pth"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# --- Define Test Transform ---
test_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load Test Dataset ---
test_dataset = datasets.ImageFolder(root="src/data/test", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- Load Model ---
model = CombinedModel(n_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)

# --- Compute Accuracy ---
def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

# --- Evaluate ---
test_acc = compute_accuracy(model, test_loader, device)
print(f"✅ Test Accuracy: {test_acc:.2f}%")
