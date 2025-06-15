import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import numpy as np

# --- Configurações ---
BASE_DIR = r"C:\Users\pedro\OneDrive\Documentos\UNIP TCC\Novo arquivo de treinamento para teste"

IMAGE_SIZE = (64, 64)
IMAGE_CHANNELS = 3
INPUT_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_CHANNELS)


CLASS_NAMES = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])
NUM_CLASSES = len(CLASS_NAMES) 

BATCH_SIZE = 32
EPOCHS = 20
MODEL_SAVE_PATH = 'modelo_libras_ABC.h5'

print(f"Diretório base das imagens: {BASE_DIR}")
print(f"Classes detectadas: {CLASS_NAMES}")
print(f"Número de classes: {NUM_CLASSES}")


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMAGE_SIZE,
    color_mode='rgb' if IMAGE_CHANNELS == 3 else 'grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    classes=CLASS_NAMES 
)

validation_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMAGE_SIZE,
    color_mode='rgb' if IMAGE_CHANNELS == 3 else 'grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    classes=CLASS_NAMES 
)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),


    Dense(NUM_CLASSES, activation='softmax')
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\nIniciando o treinamento do modelo...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

print("\nTreinamento concluído. Avaliando o modelo...")
loss, accuracy = model.evaluate(validation_generator)
print(f"Precisão do modelo no conjunto de validação: {accuracy*100:.2f}%")

model.save(MODEL_SAVE_PATH)
print(f"Modelo salvo em: {MODEL_SAVE_PATH}")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy de Treinamento')
plt.plot(history.history['val_accuracy'], label='Accuracy de Validação')
plt.title('Curva de Acurácia')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss de Treinamento')
plt.plot(history.history['val_loss'], label='Loss de Validação')
plt.title('Curva de Loss')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.show()
