import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Caminho base
CAMINHO_BASE = "C:\\Users\\pedro\\Downloads\\archive\\asl_alphabet_train\\asl_alphabet_train"
TAMANHO_IMAGEM = (64, 64)
EPOCAS = 30
BATCH_SIZE = 32
MAX_SAMPLES = 1000  # reduzir se necessário por performance

def load_all_letters(base_dir, max_samples=MAX_SAMPLES):
    images = []

    labels = []
    class_names = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isalpha() and len(d)==1])
    label_dict = {char: idx for idx, char in enumerate(class_names)}

    print(f"Classes encontradas: {label_dict}")

    for letter in class_names:
        letter_dir = os.path.join(base_dir, letter)
        count = 0
        for i, file in enumerate(os.listdir(letter_dir)):
            if count >= max_samples:
                break
            if file.endswith('.jpg'):
                img_path = os.path.join(letter_dir, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, TAMANHO_IMAGEM)
                    images.append(img)
                    labels.append(label_dict[letter])
                    count += 1
        print(f"Carregados {count} exemplos de {letter}")
    
    return np.array(images), np.array(labels), class_names

X, y, class_names = load_all_letters(CAMINHO_BASE)
X = X.astype('float32') / 255.0
y_cat = to_categorical(y, num_classes=len(class_names))

# Visualizar algumas imagens
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X[i])
    plt.title(class_names[y[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()

X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, stratify=y)


model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')  # 26 classes
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Treinamento
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCAS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# Salvar o modelo
model.save('modelo_libras_AZ.h5')
print("\n✅ Modelo salvo como 'modelo_libras_AZ.h5'")
