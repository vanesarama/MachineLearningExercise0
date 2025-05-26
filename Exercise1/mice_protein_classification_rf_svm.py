# %% [markdown]
# # Mice Protein Expression Classification (Random Forest and SVM)
# This notebook implements classification on the **Mice Protein Expression** dataset (UCI) for the VU Machine Learning 2025S Exercise 1, using only **Random Forest** and **SVM** classifiers. The dataset has 1,080 instances, 77 features (protein expression levels), and 8 classes. We:
# - Load the local `Data_Cortex_Nuclear.xls` file.
# - Explore dataset characteristics (shape, missing values, class distribution).
# - Preprocess data (impute missing values, encode labels, scale features, 70-30 split).
# - Run experiments with Random Forest and SVM, testing multiple parameter settings.
# - Evaluate holdout (70-30) and 5-fold CV performance (accuracy, precision, recall, F1-score, ROC-AUC).
# - Visualize results (confusion matrix, parameter effects) and save outputs (`classification_results.csv`, figures).
# - Compare holdout vs. CV and analyze runtime for the report.

# %% [markdown]
# ## Step 1: Import Libraries
# Import required libraries and set up the output directory.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import uuid
%matplotlib inline

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
output_dir = "./output_mice_rf_svm"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created at: {output_dir}")

# %% [markdown]
# ## Step 2: Load and Explore Dataset
# Load the Mice Protein Expression dataset from the local file `Data_Cortex_Nuclear.xls` and explore its characteristics. Saves visualizations (`class_distribution.png`, `feature_distribution.png`) for the report.

# %%
def load_dataset():
    """Load and explore the Mice Protein Expression dataset from local file."""
    file_path = "./Data_Cortex_Nuclear.xls"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file '{file_path}' not found in the current directory. Please ensure 'Data_Cortex_Nuclear.xls' is in the root folder (/Users/van/Desktop/testi).")
    
    try:
        df = pd.read_excel(file_path)
    except ImportError as e:
        if "xlrd" in str(e):
            print("Error: 'xlrd' library is missing. Install it using 'pip install xlrd>=2.0.1'.")
            raise
        elif "openpyxl" in str(e):
            print("Error: 'openpyxl' library is missing. Install it using 'pip install openpyxl'.")
            raise
        else:
            print(f"ImportError occurred: {e}")
            raise
    except Exception as e:
        print(f"Error loading dataset from '{file_path}': {e}")
        raise
    
    print("\nDataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    
    missing_values = df.isnull().sum()
    print("\nMissing Values:\n", missing_values[missing_values > 0])
    
    class_counts = df['class'].value_counts()
    print("\nClass Distribution:\n", class_counts)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Instances")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.show()
    
    plt.figure(figsize=(15, 8))
    for i, col in enumerate(df.iloc[:, 1:6].columns):
        plt.subplot(2, 3, i+1)
        sns.histplot(df[col].dropna(), bins=20, kde=True)
        plt.title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_distribution.png"))
    plt.show()
    
    return df

try:
    df = load_dataset()
except Exception as e:
    print(f"Failed to load dataset: {e}")
    raise

# %% [markdown]
# ## Step 3: Preprocess Data
# Preprocess the dataset: impute missing values, encode labels, scale features, and split data (70% train, 30% test with stratification).

# %%
def preprocess_data(df):
    """Handle missing values, encode labels, scale features, and split data."""
    # Select features (77 protein expression levels) and target
    feature_cols = [col for col in df.columns if col not in ['MouseID', 'Genotype', 'Treatment', 'Behavior', 'class']]
    X = df[feature_cols].values
    y = df['class'].values
    
    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    print("\nImputed missing values with mean. Missing values after imputation:", np.isnan(X).sum())
    
    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(y)
    class_names = le.classes_
    print("\nEncoded Classes:", class_names)
    
    # Split data (70% train, 30% test, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\nTraining Samples: {X_train.shape[0]}, Testing Samples: {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("\nScaled features using StandardScaler.")
    
    return X_train, X_test, y_train, y_test, class_names

try:
    X_train, X_test, y_train, y_test, class_names = preprocess_data(df)
except Exception as e:
    print(f"Error in preprocessing: {e}")
    raise

# %% [markdown]
# ## Step 4: Run Classification Experiments
# Run experiments with Random Forest and SVM, testing multiple parameter settings. Evaluates holdout (70-30) and 5-fold CV performance (accuracy, precision, recall, F1-score, ROC-AUC) and tracks runtime.

# %%
def run_experiments(X_train, X_test, y_train, y_test, class_names):
    """Run Random Forest and SVM experiments with multiple parameters."""
    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {'accuracy': 'accuracy', 'precision': 'precision_weighted', 'recall': 'recall_weighted', 'f1': 'f1_weighted'}
    
    # Random Forest Parameters (from your code)
    rf_params = [
        {'n_estimators': n, 'max_depth': d, 'min_samples_split': s, 'max_features': f, 'min_samples_leaf': l}
        for n in [50, 100, 200]
        for d in [None, 10, 20]
        for s in [2, 5]
        for f in ['sqrt', 'log2']
        for l in [1, 2]
    ]
    
    # SVM Parameters
    svm_params = [
        {'kernel': 'linear', 'C': 1},
        {'kernel': 'linear', 'C': 10},
        {'kernel': 'rbf', 'C': 1, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 1, 'gamma': 0.01},
        {'kernel': 'rbf', 'C': 10, 'gamma': 0.01}
    ]
    
    # Random Forest Experiments
    print("\n=== Random Forest Experiments ===")
    for param in rf_params:
        print(f"\nTesting RF: {param}")
        start_time = time.time()
        rf = RandomForestClassifier(**param, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        start_time = time.time()
        pred = rf.predict(X_test)
        prob = rf.predict_proba(X_test)
        pred_time = time.time() - start_time
        
        cv_results = cross_validate(rf, X_train, y_train, cv=kf, scoring=scoring)
        results.append({
            'Model': 'RandomForest',
            'Parameters': param,
            'Holdout_Accuracy': accuracy_score(y_test, pred),
            'Holdout_Precision': precision_score(y_test, pred, average='weighted'),
            'Holdout_Recall': recall_score(y_test, pred, average='weighted'),
            'Holdout_F1': f1_score(y_test, pred, average='weighted'),
            'Holdout_ROC_AUC': roc_auc_score(y_test, prob, multi_class='ovr'),
            'CV_Accuracy': np.mean(cv_results['test_accuracy']),
            'CV_Precision': np.mean(cv_results['test_precision']),
            'CV_Recall': np.mean(cv_results['test_recall']),
            'CV_F1': np.mean(cv_results['test_f1']),
            'CV_Accuracy_Std': np.std(cv_results['test_accuracy']),
            'Training_Time': train_time,
            'Prediction_Time': pred_time
        })
        print(f"Accuracy: {results[-1]['Holdout_Accuracy']:.2f}, F1: {results[-1]['Holdout_F1']:.2f}, CV Accuracy: {results[-1]['CV_Accuracy']:.2f} ± {results[-1]['CV_Accuracy_Std']:.2f}")
    
    # SVM Experiments
    print("\n=== SVM Experiments ===")
    for param in svm_params:
        print(f"\nTesting SVM: {param}")
        start_time = time.time()
        svm = SVC(**param, probability=True, random_state=42)
        svm.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        start_time = time.time()
        pred = svm.predict(X_test)
        prob = svm.predict_proba(X_test)
        pred_time = time.time() - start_time
        
        cv_results = cross_validate(svm, X_train, y_train, cv=kf, scoring=scoring)
        results.append({
            'Model': 'SVM',
            'Parameters': param,
            'Holdout_Accuracy': accuracy_score(y_test, pred),
            'Holdout_Precision': precision_score(y_test, pred, average='weighted'),
            'Holdout_Recall': recall_score(y_test, pred, average='weighted'),
            'Holdout_F1': f1_score(y_test, pred, average='weighted'),
            'Holdout_ROC_AUC': roc_auc_score(y_test, prob, multi_class='ovr'),
            'CV_Accuracy': np.mean(cv_results['test_accuracy']),
            'CV_Precision': np.mean(cv_results['test_precision']),
            'CV_Recall': np.mean(cv_results['test_recall']),
            'CV_F1': np.mean(cv_results['test_f1']),
            'CV_Accuracy_Std': np.std(cv_results['test_accuracy']),
            'Training_Time': train_time,
            'Prediction_Time': pred_time
        })
        print(f"Accuracy: {results[-1]['Holdout_Accuracy']:.2f}, F1: {results[-1]['Holdout_F1']:.2f}, CV Accuracy: {results[-1]['CV_Accuracy']:.2f} ± {results[-1]['CV_Accuracy_Std']:.2f}")
    
    return results, class_names

try:
    results, class_names = run_experiments(X_train, X_test, y_train, y_test, class_names)
except Exception as e:
    print(f"Error in experiments: {e}")
    raise

# %% [markdown]
# ## Step 5: Visualize and Analyze Results
# Visualize results (confusion matrix, parameter effects), save to CSV (`classification_results.csv`), and train the best model for final evaluation.

# %%
def visualize_and_save_results(results, X_train, y_train, X_test, y_test, class_names):
    """Visualize results, save to CSV, and evaluate the best model."""
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    print("\nSummary of Results:\n", results_df[['Model', 'Parameters', 'Holdout_Accuracy', 'CV_Accuracy', 'Holdout_F1', 'Training_Time']])
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, "classification_results.csv"), index=False)
    print(f"\nResults saved to {os.path.join(output_dir, 'classification_results.csv')}")
    
    # Find best model (by holdout accuracy)
    best_idx = results_df['Holdout_Accuracy'].idxmax()
    best_model = results_df.loc[best_idx]
    print(f"\nBest Model: {best_model['Model']}, Parameters: {best_model['Parameters']}")
    print(f"Holdout Accuracy: {best_model['Holdout_Accuracy']:.4f}, CV Accuracy: {best_model['CV_Accuracy']:.4f}")
    
    # Train best model
    if best_model['Model'] == 'RandomForest':
        max_depth_value = best_model['Parameters']['max_depth']
        if pd.isna(max_depth_value):
            max_depth_value = None
        else:
            max_depth_value = int(max_depth_value) if max_depth_value is not None else None
        clf = RandomForestClassifier(
            n_estimators=int(best_model['Parameters']['n_estimators']),
            max_depth=max_depth_value,
            min_samples_split=int(best_model['Parameters']['min_samples_split']),
            max_features=best_model['Parameters']['max_features'],
            min_samples_leaf=int(best_model['Parameters']['min_samples_leaf']),
            random_state=42,
            class_weight='balanced'
        )
    else:  # SVM
        clf = SVC(
            **best_model['Parameters'],
            probability=True,
            random_state=42
        )
    
    # Train and evaluate best model
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    predictions = clf.predict(X_test)
    prob = clf.predict_proba(X_test)
    pred_time = time.time() - start_time
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Best {best_model["Model"]} Confusion Matrix')
    plt.savefig(os.path.join(output_dir, "confusion_matrix_best.png"))
    plt.show()
    
    # Classification report
    print("\nClassification Report for Best Model:")
    print(classification_report(y_test, predictions, target_names=class_names))
    
    # Final metrics
    final_metrics = {
        'Classifier': best_model['Model'],
        'Holdout_Accuracy': accuracy_score(y_test, predictions),
        'Holdout_Precision': precision_score(y_test, predictions, average='weighted'),
        'Holdout_Recall': recall_score(y_test, predictions, average='weighted'),
        'Holdout_F1': f1_score(y_test, predictions, average='weighted'),
        'Holdout_ROC_AUC': roc_auc_score(y_test, prob, multi_class='ovr'),
        'Training_Time': train_time,
        'Prediction_Time': pred_time
    }
    print("\nFinal Metrics for Best Model:")
    print(pd.DataFrame([final_metrics]))
    
    # Plot parameter effects (Random Forest only)
    if best_model['Model'] == 'RandomForest':
        rf_results = results_df[results_df['Model'] == 'RandomForest']
        
        # Accuracy vs. n_estimators
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=rf_results, x='Parameters.str.n_estimators', y='Holdout_Accuracy', hue='Parameters.str.max_depth', style='Parameters.str.min_samples_split')
        plt.title('RF Accuracy vs. n_estimators')
        plt.xlabel('n_estimators')
        plt.ylabel('Holdout Accuracy')
        plt.savefig(os.path.join(output_dir, "rf_accuracy_vs_estimators.png"))
        plt.show()
        
        # Accuracy vs. max_depth
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=rf_results, x='Parameters.str.max_depth', y='Holdout_Accuracy', hue='Parameters.str.n_estimators', style='Parameters.str.min_samples_split')
        plt.title('RF Accuracy vs. max_depth')
        plt.xlabel('max_depth')
        plt.ylabel('Holdout Accuracy')
        plt.savefig(os.path.join(output_dir, "rf_accuracy_vs_max_depth.png"))
        plt.show()
        
        # Accuracy vs. max_features
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=rf_results, x='Parameters.str.max_features', y='Holdout_Accuracy', hue='Parameters.str.n_estimators', style='Parameters.str.min_samples_leaf')
        plt.title('RF Accuracy vs. max_features')
        plt.xlabel('max_features')
        plt.ylabel('Holdout Accuracy')
        plt.savefig(os.path.join(output_dir, "rf_accuracy_vs_max_features.png"))
        plt.show()
        
        # F1-score vs. n_estimators
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=rf_results, x='Parameters.str.n_estimators', y='Holdout_F1', hue='Parameters.str.max_depth', style='Parameters.str.min_samples_split')
        plt.title('RF F1-Score vs. n_estimators')
        plt.xlabel('n_estimators')
        plt.ylabel('Holdout F1-Score')
        plt.savefig(os.path.join(output_dir, "rf_f1_vs_estimators.png"))
        plt.show()
        
        # F1-score vs. max_depth
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=rf_results, x='Parameters.str.max_depth', y='Holdout_F1', hue='Parameters.str.n_estimators', style='Parameters.str.min_samples_split')
        plt.title('RF F1-Score vs. max_depth')
        plt.xlabel('max_depth')
        plt.ylabel('Holdout F1-Score')
        plt.savefig(os.path.join(output_dir, "rf_f1_vs_max_depth.png"))
        plt.show()
        
        # F1-score vs. max_features
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=rf_results, x='Parameters.str.max_features', y='Holdout_F1', hue='Parameters.str.n_estimators', style='Parameters.str.min_samples_leaf')
        plt.title('RF F1-Score vs. max_features')
        plt.xlabel('max_features')
        plt.ylabel('Holdout F1-Score')
        plt.savefig(os.path.join(output_dir, "rf_f1_vs_max_features.png"))
        plt.show()

try:
    visualize_and_save_results(results, X_train, y_train, X_test, y_test, class_names)
except Exception as e:
    print(f"Error in visualization: {e}")
    raise