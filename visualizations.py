import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from pathlib import Path


def compute_learning_curve(
    model_factory,
    X_train,
    y_train,
    X_test,
    y_test,
    train_sizes,
    seed=42
):
    """
    Compute learning curve by training on increasing subsets of data.

    Args:
        model_factory: callable that returns a NEW model instance
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        train_sizes: List of training set sizes to evaluate
        seed: Random seed for reproducibility

    Returns:
        train_accs: Array of training accuracies
        test_accs: Array of test accuracies
    """
    random.seed(seed)
    indices = list(range(len(X_train)))
    random.shuffle(indices)

    train_accs = []
    test_accs = []

    for n in train_sizes:
        print(f"\n[Learning Curve] Training with {n} samples")

        subset_idx = indices[:n]
        X_sub = [X_train[i] for i in subset_idx]
        y_sub = [y_train[i] for i in subset_idx]

        model = model_factory()
        model.fit(X_sub, y_sub)

        train_pred = model.predict(X_sub)
        test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_sub, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Train acc: {train_acc:.3f} | Test acc: {test_acc:.3f}")

    return np.array(train_accs), np.array(test_accs)


def plot_learning_curve(train_sizes, train_acc, test_acc, title, save_path=None):
    """
    Plot learning curve showing train and test accuracy vs training set size.

    Args:
        train_sizes: List of training set sizes
        train_acc: Array of training accuracies
        test_acc: Array of test accuracies
        title: Plot title
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_acc, marker='o', linewidth=2, label="Training accuracy")
    plt.plot(train_sizes, test_acc, marker='o', linewidth=2, label="Test accuracy")
    plt.xlabel("Number of training samples", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Learning curve saved to {save_path}")

    # plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, figsize=(12, 10)):
    """
    Plot confusion matrix with class names.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    # plt.show()
    return cm


def plot_per_class_metrics(y_true, y_pred, class_names, save_path=None):
    """
    Plot per-class precision, recall, and F1-score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the figure
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Extract metrics for each class
    classes = class_names
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1_score = [report[c]['f1-score'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(12, len(classes) * 0.5), 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class metrics saved to {save_path}")

    # plt.show()


def visualize_misclassifications(X, y_true, y_pred, class_names, idx_to_show=None,
                                  max_samples=20, save_path=None, confidences=None):
    """
    Visualize misclassified samples.

    Args:
        X: Input images (list of tensors or numpy arrays)
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (indexed by label)
        idx_to_show: Specific indices to show (optional)
        max_samples: Maximum number of samples to show
        save_path: Optional path to save the figure
        confidences: Optional confidence scores for predictions
    """
    # Find misclassified samples
    if idx_to_show is None:
        misclassified_idx = np.where(np.array(y_true) != np.array(y_pred))[0]
        if len(misclassified_idx) == 0:
            print("No misclassifications found!")
            return
        idx_to_show = misclassified_idx[:max_samples]

    n_samples = len(idx_to_show)
    cols = min(5, n_samples)
    rows = (n_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if n_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes

    for i, idx in enumerate(idx_to_show):
        if i >= len(axes):
            break

        # Get image
        img = X[idx]
        if hasattr(img, 'numpy'):
            img = img.numpy()
        if img.ndim == 3:
            img = img.squeeze()

        # Plot
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

        # Text with true and predicted labels (anchored below each image)
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]

        title = f"True: {true_label}\nPred: {pred_label}"
        if confidences is not None:
            title += f"\nConf: {confidences[idx]:.2%}"

        axes[i].text(
            0.5,
            -0.12,
            title,
            transform=axes[i].transAxes,
            ha='center',
            va='top',
            fontsize=10,
            color='red' if y_true[idx] != y_pred[idx] else 'green'
        )

    # Hide extra subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Misclassified Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Misclassifications visualization saved to {save_path}")

    # plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss, accuracy, learning rate).

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'val_acc', 'learning_rate'
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot training and validation loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot validation accuracy
    if 'val_acc' in history and history['val_acc']:
        axes[1].plot(epochs, [acc * 100 for acc in history['val_acc']], 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim([0, 105])

    # Plot learning rate
    if 'learning_rate' in history and history['learning_rate']:
        axes[2].plot(epochs, history['learning_rate'], 'm-', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Learning Rate', fontsize=12)
        axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[2].set_yscale('log')
        axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")

    # plt.show()


def save_predictions(X, y_true, y_pred, class_names, save_path, confidences=None):
    """
    Save predictions to a file for later analysis.

    Args:
        X: Input images (not saved, only indices)
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save predictions
        confidences: Optional confidence scores
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        f.write("Index,True_Label,True_Char,Pred_Label,Pred_Char,Correct")
        if confidences is not None:
            f.write(",Confidence")
        f.write("\n")

        for i in range(len(y_true)):
            true_char = class_names[y_true[i]]
            pred_char = class_names[y_pred[i]]
            correct = y_true[i] == y_pred[i]

            line = f"{i},{y_true[i]},{true_char},{y_pred[i]},{pred_char},{correct}"
            if confidences is not None:
                line += f",{confidences[i]:.4f}"
            f.write(line + "\n")

    print(f"Predictions saved to {save_path}")


def plot_method_comparison(results, save_path=None):
    """
    Plot separate figures for OCR methods to be used in reports.
    
    Args:
        results: List of dictionaries with keys:
                 'Model', 'Accuracy', 'Train Time (s)', 'Latency (ms/img)'
        save_path: Base path to save figures (e.g., 'plots/experiment.png').
                   The function will append suffixes like '_accuracy.png', 
                   '_latency.png', etc. to this base name.
    """
    if not results:
        print("No results to plot")
        return

    # Extract data
    models = [r['Model'] for r in results]
    accuracies = [r['Accuracy'] * 100 for r in results]
    train_times = [r['Train Time (s)'] for r in results]
    latencies = [r['Latency (ms/img)'] for r in results]

    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    # Helper to handle saving and showing
    def finalize_plot(suffix_name):
        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            # Create new filename: original_name + suffix + extension
            new_name = f"{p.stem}_{suffix_name}{p.suffix}"
            full_path = p.parent / new_name
            
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(full_path, dpi=300, bbox_inches='tight') # High DPI for LaTeX
            print(f"Saved: {full_path}")
        # plt.show()
        plt.close() # Free memory

    # --- 1. Accuracy Bar Chart ---
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(len(models)), accuracies, color=colors, edgecolor='black', alpha=0.8)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.xticks(range(len(models)), models, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0, 105])
    plt.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    finalize_plot('accuracy')

    # --- 2. Training Time Bar Chart ---
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(len(models)), train_times, color=colors, edgecolor='black', alpha=0.8)
    plt.ylabel('Training Time (seconds)', fontsize=14)
    plt.title('Training Time Comparison', fontsize=16, fontweight='bold')
    plt.xticks(range(len(models)), models, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    for bar, time in zip(bars, train_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{time:.1f}s', ha='center', va='bottom', fontsize=11)
                 
    finalize_plot('train_time')

    # --- 3. Latency Bar Chart ---
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(len(models)), latencies, color=colors, edgecolor='black', alpha=0.8)
    plt.ylabel('Latency (ms/image)', fontsize=14)
    plt.title('Inference Latency Comparison', fontsize=16, fontweight='bold')
    plt.xticks(range(len(models)), models, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    for bar, lat in zip(bars, latencies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{lat:.2f}', ha='center', va='bottom', fontsize=11)
                 
    finalize_plot('latency')

    # --- 4. Accuracy vs Training Time Scatter ---
    plt.figure(figsize=(8, 6))
    for i, (model, acc, time) in enumerate(zip(models, accuracies, train_times)):
        plt.scatter(time, acc, s=300, c=[colors[i]], edgecolors='black', linewidth=2, alpha=0.8, label=model)
    
    plt.xlabel('Training Time (seconds)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Accuracy vs Training Time', fontsize=16, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    finalize_plot('scatter_acc_time')

    # --- 5. Accuracy vs Latency Scatter ---
    plt.figure(figsize=(8, 6))
    for i, (model, acc, lat) in enumerate(zip(models, accuracies, latencies)):
        plt.scatter(lat, acc, s=300, c=[colors[i]], edgecolors='black', linewidth=2, alpha=0.8, label=model)
        
    plt.xlabel('Latency (ms/image)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Accuracy vs Inference Latency', fontsize=16, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    finalize_plot('scatter_acc_latency')

    # --- 6. Summary Table (Image) ---
    # Note: For LaTeX, it is usually better to print the LaTeX code directly (see below)
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    
    table_data = []
    # Sort for table
    sorted_results = sorted(results, key=lambda x: x['Accuracy'], reverse=True)
    
    for r in sorted_results:
        table_data.append([
            r['Model'],
            f"{r['Accuracy']*100:.2f}%",
            f"{r['Train Time (s)']:.1f}s",
            f"{r['Latency (ms/img)']:.2f}ms"
        ])

    table = plt.table(
        cellText=table_data,
        colLabels=['Model', 'Accuracy', 'Train Time', 'Latency'],
        cellLoc='center',
        loc='center',
        colColours=['lightgray'] * 4
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    plt.title('Model Performance Summary', fontsize=16, fontweight='bold')
    
    finalize_plot('table')

    # --- BONUS: Print LaTeX Table Code ---
    print("\n" + "="*30)
    print("LaTeX Table Code (Copy/Paste this into your report):")
    print("="*30)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{|l|c|c|c|}")
    print(r"\hline")
    print(r"\textbf{Model} & \textbf{Accuracy} & \textbf{Train Time (s)} & \textbf{Latency (ms)} \\")
    print(r"\hline")
    for r in sorted_results:
        print(f"{r['Model']} & {r['Accuracy']*100:.1f}\\% & {r['Train Time (s)']:.1f} & {r['Latency (ms/img)']:.2f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Comparison of OCR models sorted by accuracy.}")
    print(r"\label{tab:ocr_results}")
    print(r"\end{table}")
    print("="*30 + "\n")

def plot_feature_importance(model, feature_names=None, top_k=20, save_path=None):
    """
    Plot feature importance for models that support it (e.g., SVM with linear kernel).

    Args:
        model: Trained model with coef_ attribute
        feature_names: Optional list of feature names
        top_k: Number of top features to display
        save_path: Optional path to save the figure
    """
    if not hasattr(model, 'classifier') or not hasattr(model.classifier, 'coef_'):
        print("Model does not support feature importance visualization")
        return

    coef = model.classifier.coef_

    # For multi-class, average across classes
    if coef.ndim > 1:
        importance = np.abs(coef).mean(axis=0)
    else:
        importance = np.abs(coef)

    # Get top k features
    top_indices = np.argsort(importance)[-top_k:][::-1]
    top_importance = importance[top_indices]

    if feature_names is None:
        feature_names = [f'Feature {i}' for i in top_indices]
    else:
        feature_names = [feature_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_indices)), top_importance[::-1], color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels(feature_names[::-1])
    ax.set_xlabel('Feature Importance (|coefficient|)', fontsize=12)
    ax.set_title(f'Top {top_k} Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")

    # plt.show()
