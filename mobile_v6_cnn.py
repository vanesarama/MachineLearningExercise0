import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session

df = pd.read_csv("train.csv")
X = df.drop(columns=["price_range"]).values
y = df["price_range"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
y_cat = to_categorical(y)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
precisions = []
recalls = []

fold = 1
for train_idx, test_idx in kfold.split(X_scaled, y):
    print(f"\n--- Fold {fold} ---")

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_cat[train_idx], y_cat[test_idx]
    y_true = y[test_idx]

    clear_session()

    model = Sequential([
        Conv1D(128, kernel_size=5, activation='relu', input_shape=(X_scaled.shape[1], 1)),
        BatchNormalization(),
        Dropout(0.3),
        
        Conv1D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Conv1D(32, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Conv1D(16, kernel_size=3, activation='relu'),  # ➕ Yeni katman
        BatchNormalization(),
        Dropout(0.3),

        GlobalMaxPooling1D(),

        Dense(128, activation='relu'),  # Genişletildi
        Dropout(0.4),
        Dense(4, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')

    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    print(classification_report(y_true, y_pred, digits=4))

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    fold += 1

print("\n--- Cross-Validation Results (5-Fold) ---")
print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Mean Precision (macro): {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Mean Recall (macro): {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
