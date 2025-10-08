import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from src.models.ensemble import CombinedModel
from src.data import make_dataloaders
from src.utils import seed_everything

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(loader, desc='train'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, loss_fn, device):
    model.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='val'):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, labels)
            val_loss += loss.item()
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return val_loss / len(loader), correct / total

def main(data_dir, epochs=10, batch_size=16, lr=1e-4, save_path='best.pth'):
    seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, classes = make_dataloaders(data_dir, batch_size)
    model = CombinedModel(n_classes=len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    best_acc = 0
    for e in range(epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)
        print(f'Epoch {e+1}: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')
        if val_acc > best_acc:
            torch.save(model.state_dict(), save_path)
            best_acc = val_acc

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--save-path', type=str, default='best.pth')
    args = p.parse_args()
    main(**vars(args))
