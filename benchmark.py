import time
import random
from sklearn.metrics import accuracy_score, classification_report
from torchvision import transforms

from dataset import NISTDataset
from hog import HOGSVM_OCR
from cnn import DeepLearningOCR

def load_shared_data(dataset_root, train_limit=2000, test_limit=500):
    print(f"\n[Data] Loading dataset from {dataset_root}...")
    
    # 1. Resize to 64x64, Convert to Tensor)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # 2. Define Partitions (Training on 0-2 for speed, Testing on same)
    # TODO - remove the partition filters to train on the full alphabet
    train_partitions = ['train_30', 'train_31', 'train_32']
    train_dataset = NISTDataset(dataset_root, partitions=train_partitions, transform=transform)
    test_dataset = NISTDataset(dataset_root, partitions=['hsf_7'], transform=transform)

    # Filter test set to only match training classes (0, 1, 2)
    trained_classes = set([label for _, label in train_dataset.samples])
    test_dataset.samples = [s for s in test_dataset.samples if s[1] in trained_classes]

    # important so this specific run is reproducible
    random.seed(42) 
    
    train_indices = list(range(len(train_dataset)))
    random.shuffle(train_indices)
    train_indices = train_indices[:train_limit]

    test_indices = list(range(len(test_dataset)))
    random.shuffle(test_indices)
    test_indices = test_indices[:test_limit]

    print(f"[Data] extracting {len(train_indices)} train and {len(test_indices)} test samples...")
    X_train = [train_dataset[i][0] for i in train_indices]
    y_train = [train_dataset[i][1] for i in train_indices]
    
    X_test = [test_dataset[i][0] for i in test_indices]
    y_test = [test_dataset[i][1] for i in test_indices]

    # Get class names for reporting
    class_names = [train_dataset.idx_to_char[i] for i in sorted(list(set(y_test)))]

    return (X_train, y_train), (X_test, y_test), class_names

def evaluate_model(name, model, data_train, data_test, class_names):
    X_train, y_train = data_train
    X_test, y_test = data_test
    
    print(f"\n{'='*20} MODEL: {name} {'='*20}")
    
    print(f"[{name}] Training...")
    start_train = time.time()
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_train
    print(f"[{name}] Training Time: {train_time:.2f}s")

    print(f"[{name}] Predicting on {len(X_test)} test images...")
    start_infer = time.time()
    
    predictions = model.predict(X_test)
    
    infer_time = time.time() - start_infer
    avg_latency = (infer_time / len(X_test)) * 1000 # ms per image
    print(f"[{name}] Inference Time: {infer_time:.2f}s ({avg_latency:.2f} ms/img)")

    acc = accuracy_score(y_test, predictions)
    print(f"[{name}] Accuracy: {acc*100:.2f}%")
    print("-" * 60)
    print(classification_report(y_test, predictions, target_names=class_names))
    
    return {
        "Model": name,
        "Accuracy": acc,
        "Train Time (s)": train_time,
        "Latency (ms/img)": avg_latency
    }

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test), class_names = load_shared_data(".", train_limit=2000, test_limit=500)
    
    # TODO - implement and compare more models here
    models_to_test = [
        ("HOG + SVM", HOGSVM_OCR()),
        ("Simple CNN", DeepLearningOCR(epochs=10, batch_size=32))
    ]
    
    results = []
    for name, model in models_to_test:
        res = evaluate_model(name, model, (X_train, y_train), (X_test, y_test), class_names)
        results.append(res)
        
    print("\n" + "="*60)
    print(f"{'Model':<15} | {'Accuracy':<10} | {'Train Time':<12} | {'Latency':<10}")
    print("-" * 60)
    for res in results:
        print(f"{res['Model']:<15} | {res['Accuracy']:.2%}   | {res['Train Time (s)']:<12.2f} | {res['Latency (ms/img)']:<10.2f}")
    print("="*60)