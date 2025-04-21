# Layer number increased, drop out removed
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Concatenate, Dropout
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

y = df["label"].values

X_text_train, X_text_test, X_add_train, X_add_test, y_train, y_test = train_test_split(
    text_padded, X_additional, y, test_size=0.2, random_state=42
)

num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='text_input')
x = Embedding(input_dim=MAX_NUM_WORDS, output_dim=128)(text_input)
x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
x = GlobalMaxPooling1D()(x)

additional_input = Input(shape=(X_add_train.shape[1],), name='additional_input')
y_layer = Dense(64, activation='relu')(additional_input)

combined = Concatenate()([x, y_layer])
z = Dense(64, activation='relu')(combined)
output = Dense(num_classes, activation='softmax')(z)

model = Model(inputs=[text_input, additional_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss=focal_loss(gamma=1.0, alpha=0.5),
              metrics=['accuracy'])

model.fit(
    [X_text_train, X_add_train], y_train_cat,
    validation_data=([X_text_test, X_add_test], y_test_cat),
    epochs=5,
    batch_size=32
)

y_pred_probs = model.predict([X_text_test, X_add_test])
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, digits=4))
