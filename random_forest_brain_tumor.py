import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

# Define data directories
data_dir = "/Users/van/Desktop/TumorData"  # Local dataset path
train_dir = os.path.join(data_dir, "Training")
test_dir = os.path.join(data_dir, "Testing")
output_dir = "/Users/van/Desktop/testi"  # Local output directory

# Step 1: Explore Dataset
def explore_dataset(dataset_path):
    """Explore dataset structure and class distribution."""
    class_counts = {}
    for class_dir in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_dir)
        if os.path.isdir(class_path):
            class_counts[class_dir] = len(os.listdir(class_path))
    
    print(f"\nDataset Exploration ({dataset_path}):")
    print(f"Classes and counts: {class_counts}")
    print(f"Total images: {sum(class_counts.values())}")
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.title(f"Class Distribution in {os.path.basename(dataset_path)}")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, f'class_distribution_{os.path.basename(dataset_path)}.png'))
    plt.close()

explore_dataset(train_dir)
explore_dataset(test_dir)

# Step 2: Load Data
def load_images_from_folder(folder_path, img_size=128):
    """Load grayscale images and labels for Random Forest using Pillow."""
    X = []
    y = []
    labels = [label for label in sorted(os.listdir(folder_path)) if os.path.isdir(os.path.join(folder_path, label))]
    label_map = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        class_folder = os.path.join(folder_path, label)
        if not os.path.isdir(class_folder):
            continue
        for file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, file)
            try:
                img = Image.open(img_path).convert('L')  # 'L' for grayscale
                img = img.resize((img_size, img_size))
                img_array = np.array(img).flatten()  # Convert to NumPy array and flatten
                X.append(img_array)
                y.append(label_map[label])
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

    X = np.array(X)
    y = np.array(y)
    return X, y, labels

# Load data with checkpointing
print("\nLoading data...")
train_data_path = os.path.join(output_dir, "train_data_rf.npz")
test_data_path = os.path.join(output_dir, "test_data_rf.npz")

if os.path.exists(train_data_path) and os.path.exists(test_data_path):
    print("Loading saved data...")
    train_data = np.load(train_data_path)
    test_data = np.load(test_data_path)
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    class_names = list(train_data['class_names'])
else:
    X_train, y_train, class_names = load_images_from_folder(train_dir)
    X_test, y_test, _ = load_images_from_folder(test_dir)
    np.savez(train_data_path, X=X_train, y=y_train, class_names=class_names)
    np.savez(test_data_path, X=X_test, y=y_test)

print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Class names: {class_names}")

# Step 3: Preprocess Data
def preprocess_data(X_train, X_test, n_components=100):
    """Scale data and apply PCA."""
    print("\nChecking for missing values...")
    if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
        print("Warning: Missing values detected. Consider imputation.")
    else:
        print("No missing values found.")
    
    print("Checking for outliers...")
    if np.any(X_train < 0) or np.any(X_train > 255) or np.any(X_test < 0) or np.any(X_test > 255):
        print("Warning: Pixel values outside [0, 255] detected.")
    else:
        print("No outliers found in pixel values.")
    
    X_train_scaled = X_train / 255.0
    X_test_scaled = X_test / 255.0
    
    print(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.2f}")
    
    return X_train_pca, X_test_pca

n_components_list = [100, 200, 300]
pca_data = {}

for n_components in n_components_list:
    pca_path = os.path.join(output_dir, f"pca_data_rf_{n_components}.npz")
    if os.path.exists(pca_path):
        print(f"Loading saved PCA data for n_components={n_components}...")
        data = np.load(pca_path)
        pca_data[n_components] = (data['X_train_pca'], data['X_test_pca'])
    else:
        print(f"\nPreprocessing with n_components={n_components}")
        X_train_pca, X_test_pca = preprocess_data(X_train, X_test, n_components=n_components)
        pca_data[n_components] = (X_train_pca, X_test_pca)
        np.savez(pca_path, X_train_pca=X_train_pca, X_test_pca=X_test_pca)

# Step 4: Random Forest Experiments
def run_rf_experiments(X_train, y_train, X_test, y_test, class_names, n_components):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [1, 2]
    }
    total_experiments = np.prod([len(v) for v in param_grid.values()])
    print(f"\nRunning {total_experiments} experiments for n_components={n_components}...")
    
    results = []
    experiment_count = 0
    
    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            for min_samples_split in param_grid['min_samples_split']:
                for max_features in param_grid['max_features']:
                    for min_samples_leaf in param_grid['min_samples_leaf']:
                        experiment_count += 1
                        print(f"Experiment {experiment_count}/{total_experiments}: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, max_features={max_features}, min_samples_leaf={min_samples_leaf}")
                        
                        rf_classifier = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            max_features=max_features,
                            min_samples_leaf=min_samples_leaf,
                            random_state=42,
                            class_weight='balanced',
                            n_jobs=-1
                        )
                        
                        start_time = time.time()
                        rf_classifier.fit(X_train, y_train)
                        training_time = time.time() - start_time
                        
                        start_time = time.time()
                        predictions = rf_classifier.predict(X_test)
                        prediction_time = time.time() - start_time
                        
                        accuracy = accuracy_score(y_test, predictions)
                        f1 = f1_score(y_test, predictions, average='weighted')
                        roc_auc = roc_auc_score(y_test, rf_classifier.predict_proba(X_test), multi_class='ovr')
                        
                        cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5, scoring='accuracy')
                        cv_accuracy = np.mean(cv_scores)
                        
                        results.append({
                            'n_components': n_components,
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'max_features': max_features,
                            'min_samples_leaf': min_samples_leaf,
                            'accuracy': accuracy,
                            'f1_score': f1,
                            'roc_auc': roc_auc,
                            'cv_accuracy': cv_accuracy,
                            'training_time': training_time,
                            'prediction_time': prediction_time
                        })
                        
                        print(f"Accuracy: {accuracy:.2f}, F1-Score: {f1:.2f}, ROC-AUC: {roc_auc:.2f}, CV Accuracy: {cv_accuracy:.2f}")
                        print(f"Training Time: {training_time:.2f}s, Prediction Time: {prediction_time:.2f}s")
    
    return results

all_results = []
results_path = os.path.join(output_dir, "rf_results_all_pca.csv")

if os.path.exists(results_path):
    print("Loading saved experiment results...")
    all_results = pd.read_csv(results_path).to_dict('records')
else:
    for n_components in n_components_list:
        X_train_pca, X_test_pca = pca_data[n_components]
        results = run_rf_experiments(X_train_pca, y_train, X_test_pca, y_test, class_names, n_components)
        all_results.extend(results)
        pd.DataFrame(all_results).to_csv(results_path, index=False)

# Step 5: Visualize and Save Results
def visualize_and_save_results(all_results, X_train, y_train, X_test, y_test, class_names):
    results_df = pd.DataFrame(all_results)
    print("\nCombined Results for All PCA Settings:")
    print(results_df)
    
    results_df.to_csv(os.path.join(output_dir, "rf_results_all_pca.csv"), index=False)
    
    best_idx = results_df['accuracy'].idxmax()
    max_depth = results_df.loc[best_idx]['max_depth']
    max_depth = None if pd.isna(max_depth) else int(max_depth)
    
    best_rf = RandomForestClassifier(
        n_estimators=int(results_df.loc[best_idx]['n_estimators']),
        max_depth=max_depth,
        min_samples_split=int(results_df.loc[best_idx]['min_samples_split']),
        max_features=results_df.loc[best_idx]['max_features'],
        min_samples_leaf=int(results_df.loc[best_idx]['min_samples_leaf']),
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    n_components = int(results_df.loc[best_idx]['n_components'])
    X_train_pca, X_test_pca = pca_data[n_components]
    best_rf.fit(X_train_pca, y_train)
    predictions = best_rf.predict(X_test_pca)
    
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Best Random Forest Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'rf_confusion_matrix_best.png'))
    plt.close()
    
    print("\nClassification Report for Best Model:")
    print(classification_report(y_test, predictions, target_names=class_names))
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='n_estimators', y='accuracy', hue='n_components', style='max_depth')
    plt.title('Accuracy vs. n_estimators by PCA Components')
    plt.savefig(os.path.join(output_dir, 'rf_accuracy_vs_estimators_pca.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='max_features', y='accuracy', hue='n_components', style='min_samples_leaf')
    plt.title('Accuracy vs. max_features by PCA Components')
    plt.savefig(os.path.join(output_dir, 'rf_accuracy_vs_max_features_pca.png'))
    plt.close()
    
    test_ids = [f"test_{i}" for i in range(len(y_test))]
    submission_df = pd.DataFrame({'id': test_ids, 'predicted_class': [class_names[p] for p in predictions]})
    submission_df.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)
    print("\nKaggle submission saved as 'submission.csv'")

visualize_and_save_results(all_results, X_train, y_train, X_test, y_test, class_names)

# Step 6: Compare Holdout vs. Cross-Validation
results_df = pd.DataFrame(all_results)
print("\nHoldout vs. Cross-Validation Comparison:")
print(results_df[['n_components', 'n_estimators', 'max_depth', 'accuracy', 'cv_accuracy']])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='accuracy', y='cv_accuracy', hue='n_components', size='n_estimators')
plt.plot([results_df['accuracy'].min(), results_df['accuracy'].max()], 
         [results_df['accuracy'].min(), results_df['accuracy'].max()], 'k--')
plt.title('Holdout Accuracy vs. Cross-Validation Accuracy')
plt.xlabel('Holdout Accuracy')
plt.ylabel('Cross-Validation Accuracy')
plt.savefig(os.path.join(output_dir, 'holdout_vs_cv_accuracy.png'))
plt.close()

results_df['accuracy_diff'] = results_df['accuracy'] - results_df['cv_accuracy']
print("\nAverage Accuracy Difference (Holdout - CV):")
print(results_df.groupby('n_components')['accuracy_diff'].mean())