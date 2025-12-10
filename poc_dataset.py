"""
POC Dataset for Image Classification
Classes: Chorionic_villi, Decidual_tissue, Hemorrhage, Trophoblastic_tissue
"""

import os
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class POCDataset(Dataset):
    def __init__(self, data_dir, data_type="Training", size=(224, 224),
                 is_augment=False, transform=None, target_transform=None):
        """
        Args:
            data_dir: Path to the POC_Dataset folder
            data_type: "Training" or "Testing"
            size: Target image size (default 224×224 for GoogLeNet)
            is_augment: Enable manual augmentation
            transform: torchvision transform for preprocessing
            target_transform: Optional transform applied to labels
        """
        super().__init__()
        self.data_dir = data_dir
        self.data_type = data_type
        self.size = size
        self.is_augment = is_augment
        self.transform = transform
        self.target_transform = target_transform
        
        # Label definitions
        self.class_names = [
            'Chorionic_villi',
            'Decidual_tissue',
            'Hemorrhage',
            'Trophoblastic_tissue'
        ]
        self.label_map = {i: name for i, name in enumerate(self.class_names)}
        self.name_to_label = {name: i for i, name in enumerate(self.class_names)}
        
        self.image_names, self.labels = self._process_data()
        
        print(f"Loaded {len(self.image_names)} images from {data_type} dataset")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        label = self.labels[idx]

        class_name = self.label_map[label]
        image_path = os.path.join(self.data_dir, self.data_type, class_name, image_name)

        image = Image.open(image_path).convert('RGB')

        # Apply torchvision transforms
        if self.transform:
            image = self.transform(image)

        # Apply label transform if provided
        if self.target_transform:
            label = self.target_transform(label)

        # Manual augmentation (optional)
        if self.is_augment and self.data_type == "Training":
            image = self._apply_augmentation(image)

        return image, torch.tensor(label, dtype=torch.long)
    
    def _process_data(self):
        """Collect image filenames and corresponding labels."""
        all_images = []
        all_labels = []
        
        for label, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.data_dir, self.data_type, class_name)
            
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} does not exist.")
                continue
            
            images = [
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
            
            all_images.extend(images)
            all_labels.extend([label] * len(images))
            
            print(f"  {class_name}: {len(images)} images")
        
        return all_images, all_labels
    
    def _get_class_distribution(self):
        """Returns the number of samples per class."""
        return {
            self.label_map[label]: self.labels.count(label)
            for label in range(len(self.class_names))
        }
    
    def _preprocess_image(self, image):
        """Optional manual preprocessing."""
        return cv2.resize(image, self.size, interpolation=cv2.INTER_CUBIC)
    
    def _apply_augmentation(self, image):
        """Additional augmentations applied to training images."""
        augmentation = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1)
        ])
        return augmentation(image)


def get_data_transforms(is_training=True):
    """Defines preprocessing transforms for GoogLeNet."""
    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


if __name__ == "__main__":
    data_dir = "POC_Dataset"
    
    train_transform = get_data_transforms(is_training=True)
    test_transform = get_data_transforms(is_training=False)
    
    train_dataset = POCDataset(data_dir, data_type="Training",
                               transform=train_transform, is_augment=False)
    test_dataset = POCDataset(data_dir, data_type="Testing",
                              transform=test_transform, is_augment=False)
    
    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    image, label = train_dataset[0]
    print(f"\nSample image shape: {image.shape}")
    print(f"Sample label: {label} ({train_dataset.label_map[label.item()]})")
