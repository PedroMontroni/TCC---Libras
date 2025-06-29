import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import numpy as np
import matplotlib.pyplot as plt
import random

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

BASE_DATASET_DIR = r"C:\Users\Master\Documents\UNIP TCC\Novo arquivo de treinamento para teste" 

IMAGE_SIZE = (128, 128)
IMAGE_CHANNELS = 3
INPUT_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_CHANNELS)

CLASS_NAMES = sorted([d for d in os.listdir(BASE_DATASET_DIR) if os.path.isdir(os.path.join(BASE_DATASET_DIR, d))])
NUM_CLASSES = len(CLASS_NAMES)

if NUM_CLASSES == 0:
    raise ValueError(f"Nenhuma subpasta (classe) encontrada em: {BASE_DATASET_DIR}. Verifique a estrutura do seu dataset.")

BATCH_SIZE = 32
EPOCHS = 50
MODEL_SAVE_PATH = 'modelo_libras_otimizado.h5'

print(f"Diretório base do dataset: {BASE_DATASET_DIR}")
print(f"Classes detectadas para treinamento: {CLASS_NAMES}")
print(f"Número de classes: {NUM_CLASSES}")
print(f"Dimensões das imagens para o modelo: {IMAGE_SIZE} com {IMAGE_CHANNELS} canais")

def setup_data_generators(base_dir, image_size, image_channels, batch_size, class_names):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=image_size,
        color_mode='rgb' if image_channels == 3 else 'grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        classes=class_names,
        seed=SEED
    )

    print("\n--- ORDEM DAS CLASSES DETECTADAS PELO GERADOR DE TREINAMENTO ---")
    print("COPIE ESTA LISTA DE NOMES NA MESMA ORDEM PARA O SEU SCRIPT DE TESTE!")
    class_indices_ordered = [k for k, v in sorted(train_generator.class_indices.items(), key=lambda item: item[1])]
    print(f"Lista para o script de teste: {class_indices_ordered}")
    print("------------------------------------------------------------------\n")

    validation_generator = datagen.flow_from_directory(
        base_dir,
        target_size=image_size,
        color_mode='rgb' if image_channels == 3 else 'grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        classes=class_names,
        seed=SEED
    )
    return train_generator, validation_generator

train_generator, validation_generator = setup_data_generators(
    BASE_DATASET_DIR, IMAGE_SIZE, IMAGE_CHANNELS, BATCH_SIZE, CLASS_NAMES
)

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape), 
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.35),

        Flatten(),

        Dense(256, activation='relu'),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])
    return model

model = build_cnn_model(INPUT_SHAPE, NUM_CLASSES)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001,
    verbose=1
)

callbacks = [early_stopping, model_checkpoint, reduce_lr]

print("\nIniciando o treinamento do modelo...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=callbacks
)

print("\nTreinamento concluído. Avaliando o melhor modelo salvo...")
best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
loss, accuracy = best_model.evaluate(validation_generator)
print(f"Precisão do MELHOR modelo no conjunto de validação: {accuracy*100:.2f}%")

print(f"O melhor modelo foi salvo em: {MODEL_SAVE_PATH}")

def plot_training_history(history):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title('Curva de Acurácia ao Longo das Épocas')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss de Treinamento')
    plt.plot(history.history['val_loss'], label='Loss de Validação')
    plt.title('Curva de Loss ao Longo das Épocas')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_training_history(history)
