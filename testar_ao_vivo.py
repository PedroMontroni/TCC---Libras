import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import os

MODEL_PATH = 'modelo_libras_ABC.h5'
model = tf.keras.models.load_model(MODEL_PATH)

BASE_DIR = r"C:\Users\pedro\OneDrive\Documentos\UNIP TCC\Novo arquivo de treinamento para teste"
CLASS_NAMES = sorted(os.listdir(BASE_DIR))

TAMANHO_IMAGEM = (64, 64)
IMAGE_CHANNELS = 3

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print(f"Modelo '{MODEL_PATH}' carregado. Classes: {CLASS_NAMES}")
print("Pressione 'ESPAÇO' para analisar a mão | 'ESC' para sair.")

analisando = False
tempo_inicio = 0
letra_prevista = "Aguardando..."

def destacar_mao(frame, hand_landmarks, x_min, y_min, x_max, y_max):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    hand_points = []
    for lm in hand_landmarks.landmark:
        px, py = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
        hand_points.append((px, py))
    if len(hand_points) > 0:
        hull = cv2.convexHull(np.array(hand_points))
        cv2.drawContours(mask, [hull], 0, 255, -1)
    roi_mask = mask[y_min:y_max, x_min:x_max]
    roi = frame[y_min:y_max, x_min:x_max]
    if roi_mask.shape == roi.shape[:2]:
        masked_roi = cv2.bitwise_and(roi, roi, mask=roi_mask)
    else:
        masked_roi = roi
    bg = np.full_like(masked_roi, 255)
    resultado = np.where(masked_roi == 0, bg, masked_roi)
    return resultado

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    processed_roi_for_debug = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            padding = 30
            x_min = int(min(x_coords) * w) - padding
            y_min = int(min(y_coords) * h) - padding
            x_max = int(max(x_coords) * w) + padding
            y_max = int(max(y_coords) * h) + padding
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, w), min(y_max, h)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            if analisando and time.time() - tempo_inicio >= 5:
                if x_max > x_min and y_max > y_min:
                    roi_destacada = destacar_mao(frame, hand_landmarks, x_min, y_min, x_max, y_max)
                    if roi_destacada.shape[0] > 0 and roi_destacada.shape[1] > 0:
                        roi_resized = cv2.resize(roi_destacada, TAMANHO_IMAGEM)
                        cv2.imwrite("debug_roi_input.png", roi_resized)
                        if IMAGE_CHANNELS == 3:
                            roi_normalized = roi_resized.astype("float32") / 255.0
                        else:
                            roi_normalized = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
                            roi_normalized = np.expand_dims(roi_normalized, axis=-1)
                        roi_input = np.expand_dims(roi_normalized, axis=0)
                        pred = model.predict(roi_input)
                        letra_prevista = CLASS_NAMES[np.argmax(pred)]
                        confianca = np.max(pred) * 100
                        tempo_processamento = round(time.time() - tempo_inicio, 2)
                        cv2.putText(frame, f"Letra: {letra_prevista} ({confianca:.2f}%)",
                                    (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,

