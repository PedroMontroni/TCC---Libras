import cv2
import mediapipe as mp
import time
import os

# Define o caminho completo onde as imagens serão salvas
# Certifique-se de que este diretório exista antes de rodar o script,
# ou o script tentará criá-lo.
SAVE_PATH = r"C:\Users\pedro\OneDrive\Documentos\UNIP TCC\Novo arquivo de treinamento para teste\F"

# Configurações do MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# Não precisamos de mp_draw se não vamos desenhar as landmarks nas imagens salvas,
# mas mantemos para a visualização na janela de preview.
mp_draw = mp.solutions.drawing_utils

# Inicializar a câmera
cap = cv2.VideoCapture(0)

# Contador para nomear as imagens
img_counter = 1 # Começa do 1 para a1, a2, a3...

# Criar a pasta se não existir
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    print(f"Pasta '{SAVE_PATH}' criada.")
else:
    print(f"Pasta '{SAVE_PATH}' já existe. As imagens serão salvas aqui.")


print("Pressione 'ESPAÇO' para tirar uma foto da mão | 'ESC' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar o frame da câmera.")
        break

    frame_display = frame.copy() # Cria uma cópia para exibição, onde desenharemos os tracks
    frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB) # Converte para RGB para o MediaPipe
    result = hands.process(frame_rgb) # Processa o frame para detectar as mãos

    # Se uma mão for detectada
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # DESENHA os landmarks da mão APENAS na cópia 'frame_display' para visualização
            mp_draw.draw_landmarks(frame_display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calcula as coordenadas da caixa delimitadora (bounding box) da mão
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            # Adiciona um padding à caixa delimitadora para garantir que toda a mão seja capturada
            padding = 30 # Ajuste este valor conforme necessário
            x_min = int(min(x_coords) * frame.shape[1]) - padding
            y_min = int(min(y_coords) * frame.shape[0]) - padding
            x_max = int(max(x_coords) * frame.shape[1]) + padding
            y_max = int(max(y_coords) * frame.shape[0]) + padding

            # Garante que as coordenadas não saiam dos limites do frame original
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, frame.shape[1]), min(y_max, frame.shape[0])

            # Desenha o retângulo verde ao redor da mão NA CÓPIA DE EXIBIÇÃO
            cv2.rectangle(frame_display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Exibe o frame (com os tracks) na janela
    cv2.imshow("Captura de Imagens para Treinamento (Libras)", frame_display)

    # Captura a tecla pressionada
    key = cv2.waitKey(1) & 0xFF

    # Se a tecla ESC for pressionada, sai do loop
    if key == 27:
        print("Saindo...")
        break
    # Se a barra de espaço for pressionada e uma mão for detectada
    elif key == 32 and result.multi_hand_landmarks:
        # Pega a primeira mão detectada para o recorte (usando o frame original, sem landmarks)
        hand_landmarks = result.multi_hand_landmarks[0]

        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]

        padding = 30
        x_min = int(min(x_coords) * frame.shape[1]) - padding
        y_min = int(min(y_coords) * frame.shape[0]) - padding
        x_max = int(max(x_coords) * frame.shape[1]) + padding
        y_max = int(max(y_coords) * frame.shape[0]) + padding

        x_min, y_min = max(x_min, 0), max(y_min, 0)
        x_max, y_max = min(x_max, frame.shape[1]), min(y_max, frame.shape[0])

        # Recorta a região de interesse (ROI) da mão DO FRAME ORIGINAL (SEM LANDMARKS)
        # Verifica se a ROI é válida antes de recortar
        if x_max > x_min and y_max > y_min:
            roi = frame[y_min:y_max, x_min:x_max]

            # Define o nome completo do arquivo com o caminho especificado e o novo formato
            img_name = os.path.join(SAVE_PATH, f"f{img_counter}.png")
            cv2.imwrite(img_name, roi) # Salva a imagem colorida e sem landmarks
            print(f"Imagem '{img_name}' salva!")
            img_counter += 1
        else:
            print("Não foi possível recortar a mão. Certifique-se de que a mão está visível.")


# Libera a câmera e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()
