# cross-validation
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

def focal_loss(gamma=1.0, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

df = pd.read_csv("healthcare_dataset.csv")
df = df.dropna(subset=["Test Results"])

columns_to_drop = [
    'Name', 'Doctor', 'Hospital', 'Room Number',
    'Date of Admission', 'Discharge Date',
    'Billing Amount', 'Insurance Provider'
]
df = df.drop(columns=columns_to_drop)

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["Test Results"])
df["text"] = df["Medical Condition"].astype(str) + " " + df["Medication"].astype(str)

MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
text_padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

numerical_cols = ["Age"]
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(df[numerical_cols])

categorical_cols = ["Gender", "Blood Type", "Admission Type"]
categorical_encoded = pd.get_dummies(df[categorical_cols].astype(str))
X_additional = np.concatenate([numerical_scaled, categorical_encoded.values], axis=1)

X_text = text_padded
X_num = X_additional
y = df["label"].values
num_classes = len(np.unique(y))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
precisions = []
recalls = []

fold = 1
for train_idx, test_idx in skf.split(X_text, y):
    print(f"\n--- Fold {fold} ---")

    X_text_train, X_text_test = X_text[train_idx], X_text[test_idx]
    X_num_train, X_num_test = X_num[train_idx], X_num[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='text_input')
    x = Embedding(input_dim=MAX_NUM_WORDS, output_dim=128)(text_input)
    x = Conv1D(filters=256, kernel_size=9, activation='relu')(x)
    x = Conv1D(filters=128, kernel_size=7, activation='relu')(x)
    x = Conv1D(filters=96, kernel_size=5, activation='relu')(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)

    additional_input = Input(shape=(X_num_train.shape[1],), name='additional_input')
    y_layer = Dense(64, activation='relu')(additional_input)

    combined = Concatenate()([x, y_layer])
    z = Dense(64, activation='relu')(combined)
    output = Dense(num_classes, activation='softmax')(z)

    model = Model(inputs=[text_input, additional_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=focal_loss(gamma=1.0, alpha=0.5),
                  metrics=['accuracy'])

    model.fit(
        [X_text_train, X_num_train], y_train_cat,
        validation_data=([X_text_test, X_num_test], y_test_cat),
        epochs=15,
        batch_size=32,
        verbose=0
    )

    y_pred_probs = model.predict([X_text_test, X_num_test])
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)

    print(f"Accuracy: {acc:.4f} | Precision (macro): {prec:.4f} | Recall (macro): {rec:.4f}")
    fold += 1

print("\n--- Cross-Validation Results (5-Fold) ---")
print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Mean Precision (macro): {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Mean Recall (macro): {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
