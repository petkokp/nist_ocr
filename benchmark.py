import time
import random
import argparse
from sklearn.metrics import accuracy_score, classification_report
from torchvision import transforms
from pathlib import Path

from dataset import NISTDataset
from cnn import DeepLearningOCR
from resnet_ocr import ResNetOCR
from classical_methods import LBPSVM_OCR, HOGSVM_OCR, ZernikeOCR, ProjectionOCR

from visualizations import (
    compute_learning_curve, plot_learning_curve,
    plot_confusion_matrix, plot_per_class_metrics,
    visualize_misclassifications, plot_training_history,
    save_predictions, plot_method_comparison
)
from config import load_config, merge_args_with_config, print_config


def evaluate_model(name, model, data_train, data_test, class_names, X_test_raw, config):
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

    # Get predictions with confidence scores if available
    if hasattr(model, 'predict') and 'return_confidence' in model.predict.__code__.co_varnames:
        predictions, confidences = model.predict(X_test, return_confidence=True)
    else:
        predictions = model.predict(X_test)
        confidences = None

    infer_time = time.time() - start_infer
    avg_latency = (infer_time / len(X_test)) * 1000 # ms per image
    print(f"[{name}] Inference Time: {infer_time:.2f}s ({avg_latency:.2f} ms/img)")

    acc = accuracy_score(y_test, predictions)
    print(f"[{name}] Accuracy: {acc*100:.2f}%")
    print("-" * 60)
    labels = list(range(len(class_names)))
    print(classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=class_names,
        zero_division=0
    ))

    # Create results directory
    results_dir = Path(config['experiment']['results_dir']) / name.replace(' ', '_')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations if enabled
    if config['visualization'].get('confusion_matrix', True):
        print(f"\n[{name}] Generating confusion matrix...")
        plot_confusion_matrix(y_test, predictions, class_names,
                            save_path=results_dir / "confusion_matrix.png")

    if config['visualization'].get('per_class_metrics', True):
        print(f"[{name}] Generating per-class metrics...")
        plot_per_class_metrics(y_test, predictions, class_names,
                             save_path=results_dir / "per_class_metrics.png")

    if config['visualization'].get('misclassifications', True):
        print(f"[{name}] Visualizing misclassifications...")
        max_samples = config['visualization'].get('max_misclassifications', 20)
        visualize_misclassifications(X_test_raw, y_test, predictions, class_names,
                                    max_samples=max_samples, confidences=confidences,
                                    save_path=results_dir / "misclassifications.png")

    # Plot training history for CNN models
    if hasattr(model, 'history') and config['visualization'].get('training_history', True):
        print(f"[{name}] Plotting training history...")
        plot_training_history(model.history, save_path=results_dir / "training_history.png")

    # Save predictions
    if config['experiment'].get('save_predictions', True):
        print(f"[{name}] Saving predictions...")
        save_predictions(X_test_raw, y_test, predictions, class_names,
                       save_path=results_dir / "predictions.csv",
                       confidences=confidences)

    # Prepare results dictionary
    results = {
        "Model": name,
        "Accuracy": acc,
        "Train Time (s)": train_time,
        "Latency (ms/img)": avg_latency,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    # Add model-specific info
    if hasattr(model, 'best_val_acc'):
        results['best_val_acc'] = model.best_val_acc
    if hasattr(model, 'best_val_loss'):
        results['best_val_loss'] = model.best_val_loss
    if hasattr(model, 'optimizer'):
        results['final_lr'] = model.optimizer.param_groups[0]['lr']

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark OCR models on NIST dataset")

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help="Path to YAML config file (overrides defaults)")

    # Dataset arguments
    parser.add_argument('--dataset-root', type=str, default=None,
                        help="Root directory containing by_class folder")
    parser.add_argument('--train-limit', type=int, default=None,
                        help="Number of training samples to use")
    parser.add_argument('--test-limit', type=int, default=None,
                        help="Number of test samples to use")
    parser.add_argument('--train-partitions', type=str, nargs='+', default=None,
                        help="Training partitions to use")
    parser.add_argument('--test-partitions', type=str, nargs='+', default=None,
                        help="Test partitions to use")
    parser.add_argument('--image-size', type=int, default=None,
                        help="Resize images to this size")

    # CNN hyperparameters
    parser.add_argument('--cnn-epochs', type=int, default=None,
                        help="Number of epochs for CNN training")
    parser.add_argument('--batch-size', type=int, default=None,
                        help="Batch size for CNN training")
    parser.add_argument('--learning-rate', type=float, default=None,
                        help="Learning rate for CNN optimizer")
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd'], default=None,
                        help="Optimizer to use for CNN")
    parser.add_argument('--no-scheduler', action='store_true',
                        help="Disable learning rate scheduler")
    parser.add_argument('--early-stopping-patience', type=int, default=None,
                        help="Early stopping patience for CNN")
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help="Directory to save model checkpoints")

    # Experiment control
    parser.add_argument('--skip-learning-curves', action='store_true',
                        help="Skip learning curve generation")
    parser.add_argument('--skip-lbp', action='store_true',
                        help="Skip LBP+SVM model")
    parser.add_argument('--skip-hog', action='store_true',
                        help="Skip HOG+SVM model")
    parser.add_argument('--skip-cnn', action='store_true',
                        help="Skip CNN model")
    parser.add_argument('--skip-resnet', action='store_true',
                        help="Skip ResNet model")
    parser.add_argument('--skip-zernike', action='store_true',
                        help="Skip Zernike+SVM model")
    parser.add_argument('--skip-projection', action='store_true',
                        help="Skip Projection+kNN model")
    parser.add_argument('--results-dir', type=str, default=None,
                        help="Directory to save results")

    # Preprocessing control
    parser.add_argument('--preprocessing', action='store_true',
                        help="Enable preprocessing for classical methods")
    parser.add_argument('--threshold-method', type=str, choices=['otsu', 'sauvola'], default=None,
                        help="Adaptive thresholding method")
    parser.add_argument('--no-deskew', action='store_true',
                        help="Disable deskewing in preprocessing")

    # Classical method hyperparameters
    parser.add_argument('--knn-neighbors', type=int, default=None,
                        help="Number of neighbors for kNN classifier")
    parser.add_argument('--zernike-degree', type=int, default=None,
                        help="Degree for Zernike moments")

    # Augmentation parameters
    parser.add_argument('--augmentation-rotation', type=float, default=None,
                        help="Rotation range in degrees for augmentation")
    parser.add_argument('--augmentation-elastic', action='store_true',
                        help="Enable elastic distortion in augmentation")

    # Visualization control
    parser.add_argument('--no-visualizations', action='store_true',
                        help="Disable all visualizations")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        print(f"Loading config from {args.config}")
        config = load_config(args.config)
        config = merge_args_with_config(args, config)
    else:
        # Try to load default config
        default_config_path = Path("default_config.yaml")
        if default_config_path.exists():
            config = load_config(default_config_path)
            config = merge_args_with_config(args, config)
        else:
            # Use defaults from args
            config = {
                'dataset': {
                    'root': args.dataset_root or '.',
                    'train_partitions': args.train_partitions or ['train_30', 'train_31', 'train_32'],
                    'test_partitions': args.test_partitions or ['hsf_7'],
                    'train_limit': args.train_limit or 2000,
                    'test_limit': args.test_limit or 500,
                    'image_size': args.image_size or 64
                },
                'cnn': {
                    'epochs': args.cnn_epochs or 10,
                    'batch_size': args.batch_size or 32,
                    'learning_rate': args.learning_rate or 0.001,
                    'optimizer': args.optimizer or 'adamw',
                    'use_scheduler': not args.no_scheduler,
                    'early_stopping_patience': args.early_stopping_patience or 5
                },
                'resnet': {
                    'epochs': args.cnn_epochs or 10,
                    'batch_size': args.batch_size or 32,
                    'learning_rate': args.learning_rate or 0.001,
                    'optimizer': args.optimizer or 'adamw',
                    'use_scheduler': not args.no_scheduler,
                    'early_stopping_patience': args.early_stopping_patience or 5
                },
                'experiment': {
                    'skip_learning_curves': args.skip_learning_curves,
                    'skip_hog': args.skip_hog,
                    'skip_cnn': args.skip_cnn,
                    'skip_resnet': args.skip_resnet,
                    'checkpoint_dir': args.checkpoint_dir or 'checkpoints',
                    'results_dir': args.results_dir or 'results',
                    'save_predictions': True,
                    'save_visualizations': not args.no_visualizations
                },
                'visualization': {
                    'confusion_matrix': not args.no_visualizations,
                    'per_class_metrics': not args.no_visualizations,
                    'misclassifications': not args.no_visualizations,
                    'max_misclassifications': 20,
                    'training_history': not args.no_visualizations
                }
            }

    # Setup results directory
    results_dir = Path(config['experiment']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print_config(config)

    # Load data
    dataset_config = config['dataset']
    print(f"\n[Data] Loading dataset from {dataset_config['root']}...")
    transform = transforms.Compose([
        transforms.Resize((dataset_config['image_size'], dataset_config['image_size'])),
        transforms.ToTensor()
    ])

    # Get selected classes if specified (for filtering to specific characters)
    selected_classes = dataset_config.get('selected_classes')
    if selected_classes:
        print(f"[Data] Filtering to {len(selected_classes)} selected classes")

    train_dataset = NISTDataset(dataset_config['root'], partitions=dataset_config['train_partitions'],
                                 transform=transform, selected_classes=selected_classes)
    test_dataset = NISTDataset(dataset_config['root'], partitions=dataset_config['test_partitions'],
                                transform=transform, selected_classes=selected_classes)

    # Filter test set to only match training classes
    trained_classes = set([label for _, label in train_dataset.samples])
    test_dataset.samples = [s for s in test_dataset.samples if s[1] in trained_classes]

    # Sample data
    random.seed(42)
    train_indices = list(range(len(train_dataset)))
    random.shuffle(train_indices)
    train_indices = train_indices[:dataset_config['train_limit']]

    test_indices = list(range(len(test_dataset)))
    random.shuffle(test_indices)
    test_indices = test_indices[:dataset_config['test_limit']]

    print(f"[Data] Extracting {len(train_indices)} train and {len(test_indices)} test samples...")
    X_train = [train_dataset[i][0] for i in train_indices]
    y_train = [train_dataset[i][1] for i in train_indices]

    X_test = [test_dataset[i][0] for i in test_indices]
    y_test = [test_dataset[i][1] for i in test_indices]

    # Get class names for reporting DONT SORT
    #class_names = [train_dataset.idx_to_char[i] for i in sorted(list(set(y_test)))]
    class_names = [train_dataset.idx_to_char[i] for i in range(len(train_dataset.idx_to_char))]

    # Prepare preprocessing config for classical methods
    preprocessing_config = config.get('preprocessing') if config.get('preprocessing', {}).get('enabled', False) else None
    # Learning curves
    if not config['experiment']['skip_learning_curves']:
        max_train = len(train_indices)
        train_sizes = [100, 300, 600, 1000, 1500, min(2000, max_train)]
        train_sizes = [s for s in train_sizes if s <= max_train]

        if not config['experiment']['skip_lbp']:
            print("\n" + "="*60)
            print("LEARNING CURVE: LBP + SVM")
            print("="*60)

            
            lbp_config = config.get('lbp', {})
            lbp_train_acc, lbp_test_acc = compute_learning_curve(
                model_factory=lambda: LBPSVM_OCR(
                    radius=lbp_config.get('radius', 2),
                    n_points=lbp_config.get('n_points'),
                    method=lbp_config.get('method', 'uniform'),
                    class_weight=lbp_config.get('class_weight', 'balanced'),
                    max_iter=lbp_config.get('max_iter', 1000),
                    preprocessing_config=preprocessing_config

                ),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                train_sizes=train_sizes
            )
            plot_learning_curve(
                train_sizes,
                lbp_train_acc,
                lbp_test_acc,
                title="Learning Curve – LBP + Linear SVM"
            )

        if not config['experiment']['skip_hog']:
            print("\n" + "="*60)
            print("LEARNING CURVE: HOG + SVM")
            print("="*60)
            hog_lc_config = config.get('hog', {})
            hog_train_acc, hog_test_acc = compute_learning_curve(
                model_factory=lambda: HOGSVM_OCR(
                    orientations=hog_lc_config.get('orientations', 9),
                    pixels_per_cell=tuple(hog_lc_config.get('pixels_per_cell', [8, 8])),
                    cells_per_block=tuple(hog_lc_config.get('cells_per_block', [2, 2])),
                    class_weight=hog_lc_config.get('class_weight', 'balanced'),
                    max_iter=hog_lc_config.get('max_iter', 1000)
                ),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                train_sizes=train_sizes
            )
            plot_learning_curve(
                train_sizes,
                hog_train_acc,
                hog_test_acc,
                title="Learning Curve – HOG + Linear SVM"
            )

        if not config['experiment'].get('skip_zernike', False):
            print("\n" + "="*60)
            print("LEARNING CURVE: Zernike + SVM")
            print("="*60)
            zernike_config = config.get('zernike', {})
            zernike_train_acc, zernike_test_acc = compute_learning_curve(
                model_factory=lambda: ZernikeOCR(
                    radius=zernike_config.get('radius', 30),
                    degree=zernike_config.get('degree', 12),
                    kernel=zernike_config.get('kernel', 'rbf'),
                    C=zernike_config.get('C', 10.0),
                    gamma=zernike_config.get('gamma', 'scale'),
                    class_weight=zernike_config.get('class_weight', 'balanced')
                ),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                train_sizes=train_sizes
            )
            plot_learning_curve(
                train_sizes,
                zernike_train_acc,
                zernike_test_acc,
                title="Learning Curve – Zernike + SVM"
            )

        if not config['experiment'].get('skip_projection', False):
            print("\n" + "="*60)
            print("LEARNING CURVE: Projection + kNN")
            print("="*60)
            projection_config = config.get('projection', {})
            projection_train_acc, projection_test_acc = compute_learning_curve(
                model_factory=lambda: ProjectionOCR(
                    n_neighbors=projection_config.get('n_neighbors', 5),
                    weights=projection_config.get('weights', 'distance'),
                    metric=projection_config.get('metric', 'euclidean'),
                    normalize=projection_config.get('normalize', True)
                ),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                train_sizes=train_sizes
            )
            plot_learning_curve(
                train_sizes,
                projection_train_acc,
                projection_test_acc,
                title="Learning Curve – Projection + kNN"
            )

        if not config['experiment'].get('skip_resnet', False):
            resnet_config = config.get('resnet', {})
            print("\n" + "="*60)
            print("LEARNING CURVE: ResNet")
            print("="*60)

            def make_resnet_model():
                return ResNetOCR(
                    num_classes=len(train_dataset.class_to_idx),
                    epochs=resnet_config.get('epochs', 20),
                    batch_size=resnet_config.get('batch_size', 64),
                    learning_rate=resnet_config.get('learning_rate', 0.001),
                    early_stopping_patience=resnet_config.get('early_stopping_patience', 5),
                    checkpoint_dir=config['experiment']['checkpoint_dir'],
                    optimizer_type=resnet_config.get('optimizer', 'adamw'),
                    use_scheduler=resnet_config.get('use_scheduler', True),
                    data_augmentation=None,
                    class_weight=resnet_config.get('class_weight', 'balanced'),
                    image_size=dataset_config['image_size'],
                    weight_decay=resnet_config.get('weight_decay', 1e-4),
                    label_smoothing=resnet_config.get('label_smoothing', 0.0)
                )

            resnet_train_acc, resnet_test_acc = compute_learning_curve(
                model_factory=make_resnet_model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                train_sizes=train_sizes
            )
            plot_learning_curve(
                train_sizes,
                resnet_train_acc,
                resnet_test_acc,
                title="Learning Curve – ResNet"
            )

        if not config['experiment']['skip_cnn']:
            cnn_config = config['cnn']

            # CNN without augmentation
            print("\n" + "="*60)
            print("LEARNING CURVE: CNN")
            print("="*60)

            def make_cnn_model():
                return DeepLearningOCR(
                    num_classes=len(train_dataset.class_to_idx),
                    epochs=cnn_config['epochs'],
                    batch_size=cnn_config['batch_size'],
                    learning_rate=cnn_config['learning_rate'],
                    early_stopping_patience=cnn_config['early_stopping_patience'],
                    checkpoint_dir=config['experiment']['checkpoint_dir'],
                    optimizer_type=cnn_config['optimizer'],
                    use_scheduler=cnn_config['use_scheduler'],
                    data_augmentation=None,
                    class_weight=cnn_config.get('class_weight', 'balanced'),
                    image_size=dataset_config['image_size'],
                    weight_decay=cnn_config.get('weight_decay', 1e-4),
                    label_smoothing=cnn_config.get('label_smoothing', 0.0)
                )

            cnn_train_acc, cnn_test_acc = compute_learning_curve(
                model_factory=make_cnn_model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                train_sizes=train_sizes
            )
            plot_learning_curve(
                train_sizes,
                cnn_train_acc,
                cnn_test_acc,
                title="Learning Curve – CNN"
            )

            # CNN with augmentation
            print("\n" + "="*60)
            print("LEARNING CURVE: CNN + Aug")
            print("="*60)

            from preprocessing.augmentation import AugmentationPipeline
            aug_pipeline = AugmentationPipeline(config.get('augmentation', {}))

            def make_cnn_aug_model():
                return DeepLearningOCR(
                    num_classes=len(train_dataset.class_to_idx),
                    epochs=cnn_config['epochs'],
                    batch_size=cnn_config['batch_size'],
                    learning_rate=cnn_config['learning_rate'],
                    early_stopping_patience=cnn_config['early_stopping_patience'],
                    checkpoint_dir=config['experiment']['checkpoint_dir'],
                    optimizer_type=cnn_config['optimizer'],
                    use_scheduler=cnn_config['use_scheduler'],
                    data_augmentation=aug_pipeline,
                    class_weight=cnn_config.get('class_weight', 'balanced'),
                    image_size=dataset_config['image_size'],
                    weight_decay=cnn_config.get('weight_decay', 1e-4),
                    label_smoothing=cnn_config.get('label_smoothing', 0.0)
                )

            cnn_aug_train_acc, cnn_aug_test_acc = compute_learning_curve(
                model_factory=make_cnn_aug_model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                train_sizes=train_sizes
            )
            plot_learning_curve(
                train_sizes,
                cnn_aug_train_acc,
                cnn_aug_test_acc,
                title="Learning Curve – CNN + Aug"
            )

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION")
    print("="*60)

    models_to_test = []

    # Classical methods
    if not config['experiment']['skip_lbp']:
        lbp_config = config.get('lbp', {})
        models_to_test.append((
            "LBP + SVM",
            LBPSVM_OCR(
                radius=lbp_config.get('radius', 2),
                n_points=lbp_config.get('n_points'),
                method=lbp_config.get('method', 'uniform'),
                class_weight=lbp_config.get('class_weight', 'balanced'),
                max_iter=lbp_config.get('max_iter', 1000),
                preprocessing_config=preprocessing_config
            ),
            {
                'class_weight': lbp_config.get('class_weight', 'balanced'),
                'max_iter': lbp_config.get('max_iter', 1000)
            }
        ))

    if not config['experiment']['skip_hog']:
        hog_config = config.get('hog', {})
        models_to_test.append((
            "HOG + SVM",
            HOGSVM_OCR(
                orientations=hog_config.get('orientations', 9),
                pixels_per_cell=tuple(hog_config.get('pixels_per_cell', [8, 8])),
                cells_per_block=tuple(hog_config.get('cells_per_block', [2, 2])),
                class_weight=hog_config.get('class_weight', 'balanced'),
                max_iter=hog_config.get('max_iter', 1000),
                preprocessing_config=preprocessing_config
            ),
            {
                'class_weight': hog_config.get('class_weight', 'balanced'),
                'max_iter': hog_config.get('max_iter', 1000)
            }
        ))

    if not config['experiment'].get('skip_zernike', False):
        zernike_config = config.get('zernike', {})
        models_to_test.append((
            "Zernike + SVM",
            ZernikeOCR(
                radius=zernike_config.get('radius', 30),
                degree=zernike_config.get('degree', 12),
                kernel=zernike_config.get('kernel', 'rbf'),
                C=zernike_config.get('C', 10.0),
                gamma=zernike_config.get('gamma', 'scale'),
                class_weight=zernike_config.get('class_weight', 'balanced'),
                preprocessing_config=preprocessing_config
            ),
            {
                'method': 'zernike',
                'radius': zernike_config.get('radius', 30),
                'degree': zernike_config.get('degree', 12),
                'class_weight': zernike_config.get('class_weight', 'balanced'),
                'preprocessing': preprocessing_config is not None
            }
        ))

    if not config['experiment'].get('skip_projection', False):
        projection_config = config.get('projection', {})
        models_to_test.append((
            "Projection + kNN",
            ProjectionOCR(
                n_neighbors=projection_config.get('n_neighbors', 5),
                weights=projection_config.get('weights', 'distance'),
                metric=projection_config.get('metric', 'euclidean'),
                normalize=projection_config.get('normalize', True),
                preprocessing_config=preprocessing_config
            ),
            {
                'method': 'projection',
                'n_neighbors': projection_config.get('n_neighbors', 5),
                'preprocessing': preprocessing_config is not None
            }
        ))

    if not config['experiment'].get('skip_resnet', False):
        resnet_config = config.get('resnet', {})

        models_to_test.append((
            "ResNet",
            ResNetOCR(
                num_classes=len(train_dataset.class_to_idx),
                epochs=resnet_config.get('epochs', 20),
                batch_size=resnet_config.get('batch_size', 64),
                learning_rate=resnet_config.get('learning_rate', 0.001),
                early_stopping_patience=resnet_config.get('early_stopping_patience', 5),
                checkpoint_dir=config['experiment']['checkpoint_dir'],
                optimizer_type=resnet_config.get('optimizer', 'adamw'),
                use_scheduler=resnet_config.get('use_scheduler', True),
                data_augmentation=None,
                class_weight=resnet_config.get('class_weight', 'balanced'),
                image_size=dataset_config['image_size'],
                weight_decay=resnet_config.get('weight_decay', 1e-4),
                label_smoothing=resnet_config.get('label_smoothing', 0.0)
            ),
            {
                'optimizer': resnet_config.get('optimizer', 'adamw'),
                'learning_rate': resnet_config.get('learning_rate', 0.001),
                'batch_size': resnet_config.get('batch_size', 64),
                'epochs': resnet_config.get('epochs', 20),
                'class_weight': resnet_config.get('class_weight', 'balanced'),
                'data_augmentation': False
            }
        ))

        from preprocessing.augmentation import AugmentationPipeline
        augmentation_pipeline = AugmentationPipeline(config.get('augmentation', {}))

        models_to_test.append((
            "ResNet + Aug",
            ResNetOCR(
                num_classes=len(train_dataset.class_to_idx),
                epochs=resnet_config.get('epochs', 20),
                batch_size=resnet_config.get('batch_size', 64),
                learning_rate=resnet_config.get('learning_rate', 0.001),
                early_stopping_patience=resnet_config.get('early_stopping_patience', 5),
                checkpoint_dir=config['experiment']['checkpoint_dir'],
                optimizer_type=resnet_config.get('optimizer', 'adamw'),
                use_scheduler=resnet_config.get('use_scheduler', True),
                data_augmentation=augmentation_pipeline,
                class_weight=resnet_config.get('class_weight', 'balanced'),
                image_size=dataset_config['image_size'],
                weight_decay=resnet_config.get('weight_decay', 1e-4),
                label_smoothing=resnet_config.get('label_smoothing', 0.0)
            ),
            {
                'optimizer': resnet_config.get('optimizer', 'adamw'),
                'learning_rate': resnet_config.get('learning_rate', 0.001),
                'batch_size': resnet_config.get('batch_size', 64),
                'epochs': resnet_config.get('epochs', 20),
                'class_weight': resnet_config.get('class_weight', 'balanced'),
                'data_augmentation': True
            }
        ))

    if not config['experiment']['skip_cnn']:
        cnn_config = config['cnn']

        # CNN without augmentation
        models_to_test.append((
            "CNN",
            DeepLearningOCR(
                num_classes=len(train_dataset.class_to_idx),
                epochs=cnn_config['epochs'],
                batch_size=cnn_config['batch_size'],
                learning_rate=cnn_config['learning_rate'],
                early_stopping_patience=cnn_config['early_stopping_patience'],
                checkpoint_dir=config['experiment']['checkpoint_dir'],
                optimizer_type=cnn_config['optimizer'],
                use_scheduler=cnn_config['use_scheduler'],
                data_augmentation=None,
                class_weight=cnn_config.get('class_weight', 'balanced'),
                image_size=dataset_config['image_size'],
                weight_decay=cnn_config.get('weight_decay', 1e-4),
                label_smoothing=cnn_config.get('label_smoothing', 0.0)
            ),
            {
                'optimizer': cnn_config['optimizer'],
                'learning_rate': cnn_config['learning_rate'],
                'batch_size': cnn_config['batch_size'],
                'epochs': cnn_config['epochs'],
                'class_weight': cnn_config.get('class_weight', 'balanced'),
                'data_augmentation': False
            }
        ))

        # CNN with augmentation
        from preprocessing.augmentation import AugmentationPipeline
        augmentation_pipeline = AugmentationPipeline(config.get('augmentation', {}))

        models_to_test.append((
            "CNN + Aug",
            DeepLearningOCR(
                num_classes=len(train_dataset.class_to_idx),
                epochs=cnn_config['epochs'],
                batch_size=cnn_config['batch_size'],
                learning_rate=cnn_config['learning_rate'],
                early_stopping_patience=cnn_config['early_stopping_patience'],
                checkpoint_dir=config['experiment']['checkpoint_dir'],
                optimizer_type=cnn_config['optimizer'],
                use_scheduler=cnn_config['use_scheduler'],
                data_augmentation=augmentation_pipeline,
                class_weight=cnn_config.get('class_weight', 'balanced'),
                image_size=dataset_config['image_size'],
                weight_decay=cnn_config.get('weight_decay', 1e-4),
                label_smoothing=cnn_config.get('label_smoothing', 0.0)
            ),
            {
                'optimizer': cnn_config['optimizer'],
                'learning_rate': cnn_config['learning_rate'],
                'batch_size': cnn_config['batch_size'],
                'epochs': cnn_config['epochs'],
                'class_weight': cnn_config.get('class_weight', 'balanced'),
                'data_augmentation': True
            }
        ))

    results = []
    for name, model, model_info in models_to_test:
        res = evaluate_model(name, model, (X_train, y_train), (X_test, y_test),
                           class_names, X_test, config)
        res.update(model_info)
        results.append(res)


    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Train Time':<12} | {'Latency':<10}")
    print("-" * 60)
    for res in results:
        print(f"{res['Model']:<20} | {res['Accuracy']:.2%}   | {res['Train Time (s)']:<12.2f} | {res['Latency (ms/img)']:<10.2f}")
    print("="*60)

    # Generate method comparison visualization if enabled
    if config['visualization'].get('comparison_plot', True) and len(results) > 1:
        comparison_path = results_dir / "method_comparison.png"
        plot_method_comparison(results, save_path=comparison_path)
