# Cross-Validation

import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix


def load_images_from_folder(folder_path, img_size=128):
    X, y = [], []
    labels = [label for label in sorted(os.listdir(folder_path)) if os.path.isdir(os.path.join(folder_path, label))]
    label_map = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        class_folder = os.path.join(folder_path, label)
        for file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(label_map[label])
            except:
                continue

    X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
    y = np.array(y)
    return X, y, label_map


def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall()]
    )
    return model

# Cross-validation
start = time.time()

data_path = "Tumor"
X, y, label_map = load_images_from_folder(data_path)
num_classes = len(label_map)
y_cat = to_categorical(y, num_classes=num_classes)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
all_scores = []

for train_index, test_index in kf.split(X, y):
    print(f"\n--- Fold {fold} ---")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_cat[train_index], y_cat[test_index]

    model = create_model((128, 128, 1), num_classes)

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )

    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}, Precision: {results[2]:.4f}, Recall: {results[3]:.4f}")
    all_scores.append(results)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    fold += 1

end = time.time()
print(f"\nAverage Results over 5 folds:")
all_scores = np.array(all_scores)
metrics = ['Loss', 'Accuracy', 'Precision', 'Recall']
for i, metric in enumerate(metrics):
    print(f"{metric}: {np.mean(all_scores[:, i]):.4f} ± {np.std(all_scores[:, i]):.4f}")

print(f"\nTotal execution time: {end - start:.2f} seconds")
