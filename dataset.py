from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class NISTDataset(Dataset):
    def __init__(self, root_dir, partitions=None, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.partitions = partitions
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_char = {}

        self._load_dataset_structure()

    def _load_dataset_structure(self):
        # 1. Identify all class folders (e.g., '30', '31', '4a')
        class_root = self.root_dir / "by_class"
        if not class_root.exists():
            raise FileNotFoundError(f"Directory not found: {class_root}")

        # Sort to ensure deterministic label ordering (30 -> 0, 31 -> 1, etc.)
        class_folders = sorted([d.name for d in class_root.iterdir() if d.is_dir()])
        
        # Create mappings: Hex Folder -> Int Index -> Real Char
        for idx, hex_code in enumerate(class_folders):
            self.class_to_idx[hex_code] = idx
            # Convert hex string (e.g., "41") to ASCII char (e.g., "A")
            try:
                self.idx_to_char[idx] = chr(int(hex_code, 16))
            except ValueError:
                self.idx_to_char[idx] = "Unknown"

        # 2. Collect image paths
        # We iterate through classes, then subfolders, checking partition rules
        print(f"Scanning dataset... (Filter: {self.partitions if self.partitions else 'All'})")
        
        for hex_code in class_folders:
            class_path = class_root / hex_code
            target_idx = self.class_to_idx[hex_code]
            
            # Go through subfolders (hsf_0, train_30, etc.)
            for subfolder in class_path.iterdir():
                if not subfolder.is_dir():
                    continue
                
                # Filter logic: if partitions are set, subfolder name must contain one of them
                # e.g., if partitions=['train'], we keep 'train_30' but skip 'hsf_0'
                if self.partitions:
                    if not any(p in subfolder.name for p in self.partitions):
                        continue
                
                # Add all PNGs in this subfolder to our list
                # Using glob is faster than os.walk for flat directories
                images = list(subfolder.glob("*.png"))
                for img_path in images:
                    self.samples.append((str(img_path), target_idx))

        print(f"Dataset loaded: {len(self.samples)} images across {len(class_folders)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)
        
        return image, label