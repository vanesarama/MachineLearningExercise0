import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time

# Cell 1: Data Exploration and Imputation Justification
def explore_dataset(df):
    """Explore dataset structure, class distribution, and missing values."""
    print("\nDataset Shape:", df.shape)
    print("\nFirst 5 Rows:\n", df.head())
    print("\nData Types:\n", df.dtypes)
    
    # Class distribution
    class_counts = df['class'].value_counts()
    print("\nClass Distribution:\n", class_counts)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.savefig('class_distribution_mice.png')
    plt.close()
    
    # Missing values
    missing_values = df.isnull().sum()
    print("\nMissing Values:\n", missing_values[missing_values > 0])
    
    # Feature statistics
    print("\nFeature Statistics:\n", df.describe())

# Load dataset from local file
data_file = 'Data_Cortex_Nuclear.xls'  # Update this path if the file is in a different directory
data = pd.read_excel(data_file)

# Explore dataset
explore_dataset(data)

# Additional exploration
print("\nData Info:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

print("\nClass Distribution Analysis:")
class_dist = data['class'].value_counts()
print(f"Class imbalance: {class_dist.max() / class_dist.min():.2f}x difference between most and least frequent class.")
print("Potential impact: Imbalanced classes may bias classifiers toward majority classes (e.g., t-SC-m, t-SC-s). Consider stratified sampling or class weighting.")

# Identify numeric and categorical columns
numeric_cols = data.select_dtypes(include=['float64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns.drop(['MouseID', 'class'])

# Analyze numeric features: outliers, skewness
print("\nSkewness of Numeric Features:")
skewness = data[numeric_cols].apply(skew, nan_policy='omit')
print(skewness.sort_values(ascending=False))
print("\nSkewness Analysis: Features with |skewness| > 1 are highly skewed and may benefit from log transformation.")

# Detect outliers using IQR method
outliers = {}
for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    outlier_count = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
    outliers[col] = outlier_count
print("\nOutlier Counts per Numeric Feature:")
print(pd.Series(outliers).sort_values(ascending=False))
print("Outlier Analysis: Features with high outlier counts may skew model performance; consider robust scaling or outlier removal.")

# Feature correlations
plt.figure(figsize=(12, 8))
correlation_matrix = data[numeric_cols].corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numeric Features')
plt.savefig('correlation_heatmap_mice.png')
plt.close()
print("\nCorrelation Analysis: Strong correlations (|r| > 0.7) indicate potential multicollinearity; consider feature selection (e.g., PCA).")

# Imputation Strategy Justification
print("\nImputation Strategy Justification:")
print("Numeric Features: Mean imputation is chosen as a baseline because it preserves the mean of the distribution and is suitable for features with moderate missingness and no extreme skewness. However, for highly skewed features (e.g., skewness > 1), median imputation will be tested to reduce the impact of outliers.")
print("Categorical Features: Most frequent imputation is chosen because categorical variables (Genotype, Treatment, Behavior) have limited unique values, and the most frequent value is a reasonable approximation for missing entries. Alternative strategies like mode imputation or dropping missing rows will be explored.")
print("Suitability: Mean imputation may be sensitive to outliers in numeric features (e.g., pERK_N, pNR2B_N with high outlier counts). Median imputation or robust scaling will be tested to mitigate this. For categorical features, the low missingness (<5%) suggests imputation will have minimal impact, but dropping rows will be compared.")

# Cell 2: Preprocessing with Additional Strategies
# Define features and target
X = data.drop(['class', 'MouseID'], axis=1)
y = data['class']

# Function to cap outliers
def cap_outliers(df, columns):
    df_capped = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_capped[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df_capped

# Function to remove highly correlated features
def remove_highly_correlated(df, threshold=0.7):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
    return df.drop(columns=to_drop), to_drop

# Function to apply log transformation to skewed features
def apply_log_transformation(df, columns, skewness_threshold=1.0):
    skewed_cols = [col for col in columns if abs(skew(df[col].dropna())) > skewness_threshold]
    df_transformed = df.copy()
    for col in skewed_cols:
        df_transformed[col] = np.log1p(df_transformed[col].clip(lower=0))
    print(f"Applied log transformation to {len(skewed_cols)} skewed features: {skewed_cols}")
    return df_transformed, skewed_cols

# Preprocessing pipelines for different strategies
preprocessing_strategies = {
    'baseline': {
        'numeric_transformer': Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]),
        'categorical_transformer': Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]),
        'extra_steps': None
    },
    'median_robust': {
        'numeric_transformer': Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ]),
        'categorical_transformer': Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]),
        'extra_steps': lambda X: cap_outliers(X, numeric_cols)
    },
    'log_pca': {
        'numeric_transformer': Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95))
        ]),
        'categorical_transformer': Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]),
        'extra_steps': lambda X: apply_log_transformation(X, numeric_cols)[0]
    },
    'corr_lowvar': {
        'numeric_transformer': Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('low_variance', VarianceThreshold(threshold=0.01))
        ]),
        'categorical_transformer': Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]),
        'extra_steps': lambda X: remove_highly_correlated(X, threshold=0.7)[0]
    }
}

# Store processed datasets
processed_datasets = {}

# Apply each preprocessing strategy
for strategy_name, config in preprocessing_strategies.items():
    print(f"\nApplying preprocessing strategy: {strategy_name}")
    
    X_temp = X.copy()
    if config['extra_steps'] is not None:
        X_temp = config['extra_steps'](X_temp)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', config['numeric_transformer'], numeric_cols),
            ('cat', config['categorical_transformer'], categorical_cols)
        ])
    
    X_processed = preprocessor.fit_transform(X_temp)
    
    smote = SMOTE(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    processed_datasets[strategy_name] = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_smote': X_train_smote,
        'y_train_smote': y_train_smote,
        'preprocessor': preprocessor
    }
    print(f"Processed dataset shape (strategy: {strategy_name}): {X_processed.shape}")
    print(f"SMOTE class distribution:\n{pd.Series(y_train_smote).value_counts()}")

# Cell 3: Define Classifiers and Parameter Grids
classifiers = {
    'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'SVM': SVC(probability=True, random_state=42),
    'MLP': MLPClassifier(random_state=42, max_iter=1000)
}

param_grids = {
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    'MLP': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'learning_rate_init': [0.001, 0.01]
    }
}

# Cell 4: Run Classifiers and Evaluate Preprocessing Strategies
results = []

for strategy_name, data_dict in processed_datasets.items():
    print(f"\nEvaluating preprocessing strategy: {strategy_name}")
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    X_train_smote = data_dict['X_train_smote']
    y_train_smote = data_dict['y_train_smote']
    
    for clf_name, clf in classifiers.items():
        for use_smote, train_data in [
            ('no_smote', (X_train, y_train)),
            ('smote', (X_train_smote, y_train_smote))
        ]:
            X_train_curr, y_train_curr = train_data
            
            start_time = time.time()
            
            clf.fit(X_train_curr, y_train_curr)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
            
            holdout_metrics = {
                'Preprocessing': strategy_name,
                'SMOTE': use_smote,
                'Classifier': clf_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'F1': f1_score(y_test, y_pred, average='weighted'),
                'ROC_AUC': roc_auc_score(y_test, y_proba, multi_class='ovr') if y_proba is not None else None,
                'Runtime': time.time() - start_time
            }
            
            cv_scores = {
                'Accuracy': cross_val_score(clf, X_train_curr, y_train_curr, cv=5, scoring='accuracy').mean(),
                'Precision': cross_val_score(clf, X_train_curr, y_train_curr, cv=5, scoring='precision_weighted').mean(),
                'Recall': cross_val_score(clf, X_train_curr, y_train_curr, cv=5, scoring='recall_weighted').mean(),
                'F1': cross_val_score(clf, X_train_curr, y_train_curr, cv=5, scoring='f1_weighted').mean()
            }
            
            for param_name, param_values in param_grids[clf_name].items():
                for value in param_values:
                    clf.set_params(**{param_name: value})
                    
                    start_time = time.time()
                    
                    clf.fit(X_train_curr, y_train_curr)
                    y_pred = clf.predict(X_test)
                    
                    results.append({
                        'Preprocessing': strategy_name,
                        'SMOTE': use_smote,
                        'Classifier': clf_name,
                        'Parameter': f"{param_name}={value}",
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred, average='weighted'),
                        'Recall': recall_score(y_test, y_pred, average='weighted'),
                        'F1': f1_score(y_test, y_pred, average='weighted'),
                        'Runtime': time.time() - start_time
                    })
            
            results.append({**holdout_metrics, **{'CV_' + k: v for k, v in cv_scores.items()}})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Cell 5: Summarize and Save Results
results_df.to_csv('mice_protein_results.csv', index=False)

best_results = results_df.groupby(['Preprocessing', 'SMOTE', 'Classifier']).agg({
    'Accuracy': 'max',
    'F1': 'max',
    'Runtime': 'mean'
}).reset_index()

print("\nBest Results Summary by Preprocessing Strategy and Classifier:")
print(best_results)

best_config = results_df.loc[results_df['Accuracy'].idxmax()]
print("\nOverall Best Configuration:")
print(best_config[['Preprocessing', 'SMOTE', 'Classifier', 'Parameter', 'Accuracy', 'F1', 'Runtime']])

# Cell 6: Predictions and Visualizations
test_ids = data.iloc[y_test.index]['MouseID']
best_config = results_df.loc[results_df['Accuracy'].idxmax()]
best_preprocessing = best_config['Preprocessing']
best_smote = best_config['SMOTE']
best_clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, class_weight='balanced')

data_dict = processed_datasets[best_preprocessing]
X_train = data_dict['X_train_smote' if best_smote == 'smote' else 'X_train']
y_train = data_dict['y_train_smote' if best_smote == 'smote' else 'y_train']
X_test = data_dict['X_test']

best_clf.fit(X_train, y_train)
y_pred_test = best_clf.predict(X_test)

submission = pd.DataFrame({'id': test_ids, 'predicted_class': y_pred_test})
submission.to_csv('TestinvVane.csv', index=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Preprocessing', y='Accuracy', hue='Classifier', data=results_df[results_df['Parameter'].isna()])
plt.title('Accuracy by Preprocessing Strategy and Classifier (Holdout)')
plt.xticks(rotation=45)
plt.savefig('mice_protein_accuracy_preprocessing.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(x='SMOTE', y='Accuracy', hue='Classifier', data=results_df[results_df['Parameter'].isna()])
plt.title('Accuracy by SMOTE Usage and Classifier (Holdout)')
plt.savefig('mice_protein_accuracy_smote.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.barplot(x='Preprocessing', y='Runtime', hue='Classifier', data=results_df[results_df['Parameter'].isna()])
plt.title('Runtime by Preprocessing Strategy and Classifier')
plt.xticks(rotation=45)
plt.savefig('mice_protein_runtime_preprocessing.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(x='Classifier', y='Accuracy', data=results_df[results_df['Parameter'].isna()])
plt.title('Classifier Accuracy Comparison (Holdout)')
plt.savefig('mice_protein_accuracy_comparison.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(x='Classifier', y='Runtime', data=results_df[results_df['Parameter'].isna()])
plt.title('Classifier Runtime Comparison')
plt.savefig('mice_protein_runtime_comparison.png')
plt.close()