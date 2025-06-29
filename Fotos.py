import cv2
import mediapipe as mp
import time
import os
import numpy as np

BASE_DATASET_DIR = r"C:\Users\Master\Documents\UNIP TCC\Novo arquivo de treinamento para teste"

IMAGE_FINAL_SIZE = (128, 128)

PADDING_CAPTURE = 50 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def create_class_directory(base_dir, class_name):
    class_dir = os.path.join(base_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
        print(f"Diretório para a classe '{class_name}' criado: {class_dir}")
    else:
        print(f"Diretório para a classe '{class_name}' já existe: {class_dir}")
    return class_dir

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

if not os.path.exists(BASE_DATASET_DIR):
    os.makedirs(BASE_DATASET_DIR)
    print(f"Diretório raiz do dataset criado: {BASE_DATASET_DIR}")
else:
    print(f"Diretório raiz do dataset já existe: {BASE_DATASET_DIR}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERRO: Não foi possível abrir a câmera. Verifique se está em uso ou conectada.")
    exit()

while True:
    class_input = input("\nDigite a letra (A, B, C, D, F, G, etc.) para coletar imagens (ou 'sair' para encerrar): ").strip().upper()
    if class_input == 'SAIR':
        break
    
    if not class_input.isalpha() or len(class_input) != 1:
        print("Entrada inválida. Digite apenas uma letra (A-Z) ou 'sair'.")
        continue

    current_class_dir = create_class_directory(BASE_DATASET_DIR, class_input)
    img_counter = len(os.listdir(current_class_dir)) + 1

    print(f"\nColetando imagens para a letra: '{class_input}'")
    print(f"As imagens serão salvas em: {current_class_dir}")
    print("Posicione sua mão. Pressione 'ESPAÇO' para tirar uma foto | 'R' para resetar o contador | 'ESC' para escolher outra letra.")

    collecting = True
    while collecting:
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
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame_display, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                hand_roi, coords = preprocess_hand_image(frame, hand_landmarks, w, h, PADDING_CAPTURE)
                
                if hand_roi is not None:
                    hand_detected_and_valid = True
                    x_min_disp, y_min_disp, x_max_disp, y_max_disp = coords
                    cv2.rectangle(frame_display, (x_min_disp, y_min_disp), (x_max_disp, y_max_disp), (0, 255, 0), 2)
                    
                    cv2.putText(frame_display, f"Capturando para {class_input}: {img_counter}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(frame_display, "Mao muito perto da borda! Ajuste.", (10, h - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame_display, "Aguardando mao...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow(f"Coletando Imagens para '{class_input}'", frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:
            collecting = False
            print(f"Encerrando coleta para '{class_input}'.")
            cv2.destroyWindow(f"Coletando Imagens para '{class_input}'")
            break
        elif key == 32:
            if hand_detected_and_valid:
                final_image_to_save = cv2.resize(hand_roi, IMAGE_FINAL_SIZE, interpolation=cv2.INTER_AREA)
                img_name = os.path.join(current_class_dir, f"{class_input.lower()}{img_counter}.png")
                cv2.imwrite(img_name, final_image_to_save)
                print(f"Imagem '{img_name}' salva!")
                img_counter += 1
            else:
                print("Não foi possível salvar a imagem: Mão não detectada ou área de recorte inválida.")
        elif key == ord('r'):
            img_counter = 1
            print(f"Contador de imagens para '{class_input}' resetado para 1.")

cap.release()
cv2.destroyAllWindows()
print("\nPrograma de coleta de imagens encerrado.")
