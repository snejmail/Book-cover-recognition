from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os


class BookCoverDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def create_loaders(root_dir='archive (1)/book-covers', batch_size=64):
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = BookCoverDataset(root_dir, transform=train_transform)
    val_dataset = BookCoverDataset(root_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count() // 2 or 1,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2 or 1,
        pin_memory=True
    )

    return train_loader, val_loader


# Initialize loaders when imported
train_loader, val_loader = create_loaders()

if __name__ == '__main__':
    for images, labels in train_loader:
        print(f"Batch image shape: {images.shape}")
        print(f"Batch label shape: {labels.shape}")
        break
