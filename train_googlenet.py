"""
GoogleNet Training Pipeline for POC Dataset Classification
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import time

from poc_dataset import POCDataset, get_data_transforms



class GoogleNetTrainer:
    def __init__(self, num_classes=4, learning_rate=0.001, device=None, use_aux_loss=True):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.use_aux_loss = use_aux_loss
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cpu':
            print("Running on CPU mode")
        
        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )
        
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.best_val_acc = 0.0
    
    def _create_model(self):
        print("\nLoading model...")
        model = models.googlenet(pretrained=True, aux_logits=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        
        # Modify auxiliary classifiers
        if hasattr(model, 'aux1'):
            num_aux1 = model.aux1.fc2.in_features
            model.aux1.fc2 = nn.Linear(num_aux1, self.num_classes)
        if hasattr(model, 'aux2'):
            num_aux2 = model.aux2.fc2.in_features
            model.aux2.fc2 = nn.Linear(num_aux2, self.num_classes)
        
        model = model.to(self.device)
        print(f"Model loaded successfully (Parameters: {sum(p.numel() for p in model.parameters()):,})")
        if self.use_aux_loss:
            print("Using auxiliary loss for training")
        return model
    
    def train_one_epoch(self, dataloader, epoch_num=0):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_num} - Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            # Forward pass with auxiliary loss
            if self.model.training and self.use_aux_loss:
                outputs, aux1, aux2 = self.model(images)
                # Compute weighted loss (0.3 for auxiliary classifiers)
                loss1 = self.criterion(outputs, labels)
                loss2 = self.criterion(aux1, labels)
                loss3 = self.criterion(aux2, labels)
                
                loss = loss1 + 0.3 * loss2 + 0.3 * loss3
            else:
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
        
        return running_loss / len(dataloader), 100 * correct / total
    
    def validate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return running_loss / len(dataloader), 100 * correct / total, all_preds, all_labels
    
    def fit(self, train_loader, val_loader, num_epochs, save_dir="models"):
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Training started: {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print("-" * 60)
            
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch+1)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            val_loss, val_acc, _, _ = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            self.scheduler.step(val_loss)
            
            print(f"\nResults:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(save_dir, "best_googlenet_poc.pth"))
                print(f"  Best model saved! (Val Acc: {val_acc:.2f}%)")
            
            elapsed = (time.time() - start_time) / 60
            remaining = elapsed / (epoch + 1) * (num_epochs - epoch - 1)
            print(f"  Time elapsed: {elapsed:.0f}min, Remaining: ~{remaining:.0f}min")
        
        print(f"\n{'='*60}")
        print(f"Training completed! Best Val Acc: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")
    
    def plot_training_history(self, save_path="training_history.png"):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(self.train_losses, label='Train Loss', marker='o')
        axes[0].plot(self.val_losses, label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.train_accs, label='Train Acc', marker='o')
        axes[1].plot(self.val_accs, label='Val Acc', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Training history saved to {save_path}")
        plt.close()
    
    def evaluate_on_test(self, test_loader, class_names, save_path="confusion_matrix.png"):
        print("\nEvaluating on test set...")
        test_loss, test_acc, all_preds, all_labels = self.validate(test_loader)
        
        print(f"\n{'='*60}")
        print(f"Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.2f}%")
        print(f"{'='*60}\n")
        
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix (Test Set)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
        
        return test_acc, cm


def main():
    print("\n" + "="*60)
    print("POC Dataset - GoogleNet Training")
    print("="*60 + "\n")
    
    DATA_DIR = "POC_Dataset"
    BATCH_SIZE = 8
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001
    NUM_WORKERS = 2
    VAL_SPLIT = 0.2
    USE_AUX_LOSS = True
    
    print("Configuration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Val Split: {VAL_SPLIT*100:.0f}%")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Auxiliary Loss: {'Enabled' if USE_AUX_LOSS else 'Disabled'}\n")
    
    print("Loading datasets...")
    
    train_transform = get_data_transforms(is_training=True)
    val_transform = get_data_transforms(is_training=False)
    test_transform = get_data_transforms(is_training=False)
    
    full_train_dataset_no_aug = POCDataset(DATA_DIR, data_type="Training", 
                                           transform=None, is_augment=False)
    
    indices = list(range(len(full_train_dataset_no_aug)))
    labels = full_train_dataset_no_aug.labels
    
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=VAL_SPLIT, 
        random_state=42,
        stratify=labels
    )
    
    print(f"\nDataset split (stratified):")
    print(f"  Total training images: {len(full_train_dataset_no_aug):,}")
    print(f"  Train: {len(train_indices):,}")
    print(f"  Validation: {len(val_indices):,}")
    
    train_dataset_with_aug = POCDataset(DATA_DIR, data_type="Training", 
                                        transform=train_transform, is_augment=False)
    train_dataset = Subset(train_dataset_with_aug, train_indices)
    
    val_dataset_no_aug = POCDataset(DATA_DIR, data_type="Training", 
                                    transform=val_transform, is_augment=False)
    val_dataset = Subset(val_dataset_no_aug, val_indices)
    
    test_dataset = POCDataset(DATA_DIR, data_type="Testing", 
                             transform=test_transform, is_augment=False)
    
    print(f"  Test: {len(test_dataset):,}\n")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    
    device = torch.device("cpu")
    trainer = GoogleNetTrainer(num_classes=4, learning_rate=LEARNING_RATE, 
                              device=device, use_aux_loss=USE_AUX_LOSS)
    
    trainer.fit(train_loader, val_loader, num_epochs=NUM_EPOCHS)
    
    print("\nGenerating visualizations...")
    trainer.plot_training_history("training_history.png")
    
    class_names = full_train_dataset_no_aug.class_names
    trainer.evaluate_on_test(test_loader, class_names, "confusion_matrix.png")
    
    print("\n" + "="*60)
    print("Training completed! Generated files:")
    print("  - models/best_googlenet_poc.pth")
    print("  - training_history.png")
    print("  - confusion_matrix.png")
    print("="*60)


if __name__ == "__main__":
    main()