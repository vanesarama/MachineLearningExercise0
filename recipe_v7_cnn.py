import pandas as pd
import numpy as np
import random
import nltk
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Concatenate, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import tensorflow as tf
from tensorflow.keras import backend as K

nltk.download('wordnet')
nltk.download('omw-1.4')

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

def synonym_augment(text, n=2):
    words = text.split()
    new_words = words.copy()
    random_indices = list(range(len(words)))
    random.shuffle(random_indices)
    changed = 0
    for idx in random_indices:
        synonyms = wordnet.synsets(words[idx])
        if synonyms:
            lemmas = synonyms[0].lemma_names()
            for lemma in lemmas:
                synonym = lemma.replace("_", " ")
                if synonym.lower() != words[idx].lower():
                    new_words[idx] = synonym
                    changed += 1
                    break
        if changed >= n:
            break
    return " ".join(new_words)

df = pd.read_csv("Recipe Reviews and User Feedback Dataset.csv")
columns_to_drop = ['recipe_number', 'recipe_code', 'recipe_name', 'comment_id', 'user_id', 'user_name', 'created_at']
df_cleaned = df.drop(columns=columns_to_drop).dropna(subset=['text'])

df_01 = df_cleaned[df_cleaned['stars'].isin([0, 1])]
augmented_texts = []
augmented_labels = []

for _, row in df_01.iterrows():
    for _ in range(2):
        aug_text = synonym_augment(row['text'], n=2)
        augmented_texts.append(aug_text)
        augmented_labels.append(row['stars'])

df_augmented = pd.DataFrame({'text': augmented_texts, 'stars': augmented_labels})
df_augmented_final = pd.concat([df_cleaned, df_augmented], ignore_index=True)

texts = df_augmented_final['text'].astype(str).values
labels = df_augmented_final['stars'].values
numeric_features = df_augmented_final[['user_reputation', 'reply_count', 'thumbs_up', 'thumbs_down', 'best_score']]

scaler = StandardScaler()
numeric_features_scaled = scaler.fit_transform(numeric_features)

MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 200

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
text_padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

def oversample_class(class_label, factor):
    indices = np.where(labels == class_label)[0]
    text_part = np.repeat(text_padded[indices], factor, axis=0)
    num_part = np.repeat(numeric_features_scaled[indices], factor, axis=0)
    label_part = np.repeat(labels[indices], factor, axis=0)
    return text_part, num_part, label_part

t2, n2, l2 = oversample_class(2, 10)
t3, n3, l3 = oversample_class(3, 6)
t4, n4, l4 = oversample_class(4, 4)

X_text_all = np.concatenate([text_padded, t2, t3, t4], axis=0)
X_num_all = np.concatenate([numeric_features_scaled, n2, n3, n4], axis=0)
y_all = np.concatenate([labels, l2, l3, l4], axis=0)

X_text_all, X_num_all, y_all = shuffle(X_text_all, X_num_all, y_all, random_state=42)

X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_text_all, X_num_all, y_all, test_size=0.2, random_state=42
)

num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='text_input')
x = Embedding(input_dim=MAX_NUM_WORDS, output_dim=128)(text_input)
x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)

numeric_input = Input(shape=(X_num_train.shape[1],), name='numeric_input')
y = Dense(64, activation='relu')(numeric_input)

combined = Concatenate()([x, y])
z = Dense(64, activation='relu')(combined)
z = Dropout(0.5)(z)
z = Dense(num_classes, activation='softmax')(z)

model = Model(inputs=[text_input, numeric_input], outputs=z)

model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)

history = model.fit(
    [X_text_train, X_num_train], y_train_cat,
    validation_data=([X_text_test, X_num_test], y_test_cat),
    epochs=5,
    batch_size=32
)

y_pred_probs = model.predict([X_text_test, X_num_test])
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(num_classes))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

accuracy = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

precision = precision_score(y_true, y_pred, average='macro')
print(f"Precision (macro): {precision:.4f}")

recall = recall_score(y_true, y_pred, average='macro')
print(f"Recall (macro): {recall:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))
