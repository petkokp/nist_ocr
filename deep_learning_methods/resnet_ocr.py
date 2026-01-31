import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def _conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class SmallResNet(nn.Module):
    """
    ResNet-18 style network adapted for 1-channel, small images (e.g., 32x32).
    """
    def __init__(self, num_classes=62):
        super().__init__()
        self.in_channels = 64

        self.conv1 = _conv3x3(1, 64, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [BasicBlock(self.in_channels, out_channels, stride=stride, downsample=downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNetOCR:
    def __init__(
        self,
        num_classes=62,
        epochs=20,
        batch_size=64,
        learning_rate=0.001,
        early_stopping_patience=5,
        checkpoint_dir="checkpoints",
        optimizer_type="adamw",
        use_scheduler=True,
        data_augmentation=None,
        class_weight="balanced",
        image_size=32,
        weight_decay=1e-4,
        label_smoothing=0.0
    ):
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.data_augmentation = data_augmentation
        self.use_scheduler = use_scheduler
        self.class_weight = class_weight
        self.weight_decay = float(weight_decay) if weight_decay is not None else 1e-4
        self.label_smoothing = float(label_smoothing) if label_smoothing is not None else 0.0

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Initializing ResNet on device: {self.device}")
        print(f"Optimizer: {optimizer_type}, LR Scheduler: {use_scheduler}")
        print(f"Number of classes: {num_classes}, Image size: {image_size}")
        print(f"Weight decay: {self.weight_decay}, Label smoothing: {self.label_smoothing}")

        self.model = SmallResNet(num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        if optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        elif optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=self.weight_decay
            )

        self.scheduler = None
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3
            )
            print("Learning rate scheduler enabled (ReduceLROnPlateau: factor=0.5, patience=3)")

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.best_model_path = None

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": []
        }

    def fit(self, X, y, X_val=None, y_val=None, val_split=0.15):
        if isinstance(X, list):
            X_tensor = torch.stack(X)
        else:
            X_tensor = torch.tensor(X)

        y_tensor = torch.tensor(y)

        if X_val is None and val_split > 0:
            n_samples = len(X_tensor)
            n_val = int(n_samples * val_split)
            indices = torch.randperm(n_samples)

            train_indices = indices[n_val:]
            val_indices = indices[:n_val]

            X_train = X_tensor[train_indices]
            y_train = y_tensor[train_indices]
            X_val_tensor = X_tensor[val_indices]
            y_val_tensor = y_tensor[val_indices]

            print(f"Created validation split: {len(X_train)} train, {len(X_val_tensor)} val")
        else:
            X_train = X_tensor
            y_train = y_tensor
            if X_val is not None:
                if isinstance(X_val, list):
                    X_val_tensor = torch.stack(X_val)
                else:
                    X_val_tensor = torch.tensor(X_val)
                y_val_tensor = torch.tensor(y_val)
            else:
                X_val_tensor = None
                y_val_tensor = None

        if self.class_weight == "balanced":
            from sklearn.utils.class_weight import compute_class_weight
            y_train_np = y_train.numpy()
            classes_present = np.unique(y_train_np)
            weights = compute_class_weight("balanced", classes=classes_present, y=y_train_np)

            full_weights = np.ones(self.num_classes, dtype=np.float32)
            for cls, w in zip(classes_present, weights):
                full_weights[cls] = w

            class_weights = torch.tensor(full_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(
                "Class weighting enabled: "
                f"{len(classes_present)} classes present, "
                f"weight range: [{weights.min():.2f}, {weights.max():.2f}]"
            )

        if self.data_augmentation is not None and callable(self.data_augmentation):
            from torch.utils.data import Dataset

            class AugmentedDataset(Dataset):
                def __init__(self, X, y, augmentation):
                    self.X = X
                    self.y = y
                    self.augmentation = augmentation

                def __len__(self):
                    return len(self.X)

                def __getitem__(self, idx):
                    image = self.X[idx]
                    label = self.y[idx]
                    image = self.augmentation(image)
                    return image, label

            train_dataset = AugmentedDataset(X_train, y_train, self.data_augmentation)
            print("Data augmentation enabled for training set")
        else:
            train_dataset = TensorDataset(X_train, y_train)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if X_val_tensor is not None:
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None

        print(f"Starting training for up to {self.epochs} epochs (early stopping patience: {self.early_stopping_patience})...")

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            start_t = time.time()

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            epoch_time = time.time() - start_t

            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)

                if self.scheduler is not None:
                    self.scheduler.step(val_loss)

                current_lr = self.optimizer.param_groups[0]["lr"]
                self.history["train_loss"].append(avg_train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                self.history["learning_rate"].append(current_lr)

                print(
                    f"Epoch [{epoch+1}/{self.epochs}] "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.2%} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {epoch_time:.1f}s"
                )

                improved = False
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    improved = True

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    improved = True

                if improved:
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, val_loss, val_acc)
                    print("  -> Model improved! Checkpoint saved.")
                else:
                    self.patience_counter += 1
                    print(f"  -> No improvement ({self.patience_counter}/{self.early_stopping_patience})")

                    if self.patience_counter >= self.early_stopping_patience:
                        print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                        self._load_best_checkpoint()
                        break
            else:
                self.history["train_loss"].append(avg_train_loss)
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.history["learning_rate"].append(current_lr)
                print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {avg_train_loss:.4f} - Time: {epoch_time:.1f}s")

    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        return avg_val_loss, val_accuracy

    def _save_checkpoint(self, epoch, val_loss, val_acc):
        checkpoint_path = self.checkpoint_dir / f"best_resnet_epoch{epoch+1}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc
        }, checkpoint_path)

        if self.best_model_path and self.best_model_path.exists():
            self.best_model_path.unlink()
        self.best_model_path = checkpoint_path

    def _load_best_checkpoint(self):
        if self.best_model_path and self.best_model_path.exists():
            print(f"Loading best model from {self.best_model_path}")
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Best model: Epoch {checkpoint['epoch']+1}, "
                f"Val Loss: {checkpoint['val_loss']:.4f}, "
                f"Val Acc: {checkpoint['val_acc']:.2%}"
            )

    def predict(self, X, return_confidence=False):
        self.model.eval()

        if isinstance(X, list):
            X_tensor = torch.stack(X)
        else:
            X_tensor = torch.tensor(X)

        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_preds = []
        all_probs = []

        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                probs = torch.softmax(outputs, dim=1)
                max_probs, predicted = torch.max(probs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(max_probs.cpu().numpy())

        if return_confidence:
            return np.array(all_preds), np.array(all_probs)
        return np.array(all_preds)
