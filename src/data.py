import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def make_dataloaders(data_dir, batch_size=32, val_split=0.1, seed=42, img_size=299):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(int(img_size*1.05)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    dataset = ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
    targets = [s[1] for s in dataset.samples]
    train_idx, val_idx = train_test_split(list(range(len(targets))), test_size=val_split,
                                          stratify=targets, random_state=seed)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(ImageFolder(os.path.join(data_dir, "train"), transform=val_tfms), val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, dataset.classes
