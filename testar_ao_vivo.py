import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time

model = tf.keras.models.load_model('modelo_libras_AZ.h5')
class_names = sorted([chr(i) for i in range(65, 91)])
TAMANHO_IMAGEM = (64, 64)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print(" Pressione 'ESPAÇO' para analisar essa bosta | ESC para sair dessa bosta.")

analisando = False
tempo_inicio = 0

def destacar_mao(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    bg = np.full_like(roi, 255)
    resultado = np.where(mask[:, :, np.newaxis] == 255, roi, bg)
    return resultado

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * w) - 20
            y_min = int(min(y_coords) * h) - 20
            x_max = int(max(x_coords) * w) + 20
            y_max = int(max(y_coords) * h) + 20

            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, w), min(y_max, h)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            if analisando and time.time() - tempo_inicio >= 5:
                roi = frame[y_min:y_max, x_min:x_max]
                roi_destacada = destacar_mao(roi)

                roi_resized = cv2.resize(roi_destacada, TAMANHO_IMAGEM)

                roi_normalized = roi_resized.astype("float32") / 255.0
                roi_input = np.expand_dims(roi_normalized, axis=0)  # (1, 64, 64, 3)
                cv2.imwrite("entrada_para_modelo.png", roi_resized)


                pred = model.predict(roi_input)
                letra = class_names[np.argmax(pred)]

                tempo_processamento = round(time.time() - tempo_inicio, 2)
                cv2.putText(frame, f"Letra: {letra} ({tempo_processamento}s)",
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
                analisando = False

    if analisando:
        tempo_restante = int(5 - (time.time() - tempo_inicio))
        if tempo_restante > 0:
            cv2.putText(frame, f"Analisando essa merda em {tempo_restante}s...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Reconhecimento de Letras (Libras)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == 32 and not analisando:
        tempo_inicio = time.time()
        analisando = True

cap.release()
cv2.destroyAllWindows()
