import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import os

MODEL_PATH = 'C:\\Users\\Master\\Documents\\UNIP TCC\\new\\modelo_libras_otimizado.h5'
IMAGE_SIZE = (128, 128)
PADDING_TEST = 50

if not os.path.exists(MODEL_PATH):
    print(f"ERRO: Modelo não encontrado em '{MODEL_PATH}'. Verifique o caminho e o nome do arquivo.")
    exit()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modelo '{MODEL_PATH}' carregado com sucesso.")
except Exception as e:
    print(f"ERRO ao carregar o modelo: {e}")
    exit()

class_names = ['A', 'B', 'C', 'D', 'E', 'F']

if len(class_names) != model.layers[-1].units:
    print(f"AVISO: O número de classes definido ({len(class_names)}) não corresponde ao número de neurônios na camada de saída do modelo ({model.layers[-1].units}).")
    print("Verifique a lista 'class_names' e o seu modelo de treinamento.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def preprocess_hand_image(frame, hand_landmarks, img_width, img_height, padding):
    x_coords_px = [lm.x * img_width for lm in hand_landmarks.landmark]
    y_coords_px = [lm.y * img_height for lm in hand_landmarks.landmark]

    x_min_raw = int(min(x_coords_px))
    y_min_raw = int(min(y_coords_px))
    x_max_raw = int(max(x_coords_px))
    y_max_raw = int(max(y_coords_px))

    x_min_padded = max(0, x_min_raw - padding)
    y_min_padded = max(0, y_min_raw - padding)
    x_max_padded = min(img_width, x_max_raw + padding)
    y_max_padded = min(img_height, y_max_raw + padding)

    if x_max_padded > x_min_padded and y_max_padded > y_min_padded:
        roi = frame[y_min_padded:y_max_padded, x_min_padded:x_max_padded]
        return roi, (x_min_padded, y_min_padded, x_max_padded, y_max_padded)
    else:
        return None, None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERRO: Não foi possível abrir a câmera para o teste. Verifique conexão.")
    exit()

print("\n--- INSTRUÇÕES DO TESTE ---")
print("1. Posicione sua mão na frente da câmera dentro da caixa vermelha.")
print("2. Pressione 'ESPAÇO' para iniciar a contagem regressiva e prever a letra.")
print("3. Pressione 'ESC' para sair.")
print("---------------------------\n")

analisando = False
tempo_inicio_analise = 0
pred_letra = "N/A"
confidence = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar o frame. Tentando novamente...")
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_display = frame.copy()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    hand_detected_and_valid = False
    roi_coords = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame_display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_roi, coords = preprocess_hand_image(frame, hand_landmarks, w, h, PADDING_TEST)
            
            if hand_roi is not None:
                hand_detected_and_valid = True
                roi_coords = coords
                x_min_disp, y_min_disp, x_max_disp, y_max_disp = coords
                cv2.rectangle(frame_display, (x_min_disp, y_min_disp), (x_max_disp, y_max_disp), (0, 0, 255), 2)

                if analisando:
                    time_elapsed = time.time() - tempo_inicio_analise
                    if time_elapsed >= 3:

                        if hand_roi.shape[0] > 0 and hand_roi.shape[1] > 0:
                            roi_resized = cv2.resize(hand_roi, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                            
                            roi_normalized = roi_resized.astype("float32") / 255.0
                            
                            roi_input = np.expand_dims(roi_normalized, axis=0)

                            predictions = model.predict(roi_input)
                            predicted_class_index = np.argmax(predictions)
                            
                            pred_letra = class_names[predicted_class_index]
                            confidence = predictions[0][predicted_class_index] * 100

                            analisando = False
                        else:
                            print("AVISO: ROI capturada para previsão tem dimensões inválidas. Posicione a mão corretamente.")
                            analisando = False
            else:
                cv2.putText(frame_display, "Mao muito perto da borda! Ajuste.", (10, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
    else:
        cv2.putText(frame_display, "Aguardando mao...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(frame_display, f"Letra: {pred_letra}", (w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame_display, f"Confianca: {confidence:.2f}%", (w - 200, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    if analisando:
        time_remaining = 3 - int(time.time() - tempo_inicio_analise)
        if time_remaining > 0:
            cv2.putText(frame_display, f"Analisando em {time_remaining}s...",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(frame_display, "Processando...",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    elif not hand_detected_and_valid:
        pred_letra = "N/A"
        confidence = 0.0

    cv2.imshow("Reconhecimento de Letras (Libras)", frame_display)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == 32:
        if hand_detected_and_valid:
            analisando = True
            tempo_inicio_analise = time.time()
            pred_letra = "..."
            confidence = 0.0
        else:
            print("AVISO: Mão não detectada ou ROI inválida para iniciar a análise. Posicione a mão corretamente.")

cap.release()
cv2.destroyAllWindows()

