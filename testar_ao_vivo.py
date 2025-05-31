import cv2
import numpy as np
import tensorflow as tf
import time

model = tf.keras.models.load_model('modelo_libras_AZ.h5')
class_names = sorted([chr(i) for i in range(65, 91)])
TAMANHO_IMAGEM = (64, 64)

cap = cv2.VideoCapture(0)

print("📷 Pressione 'ESPACO' para analisar. Pressione 'ESC' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a webcam.")
        break

    frame = cv2.flip(frame, 1)

    x1, y1, x2, y2 = 200, 100, 400, 300
    roi = frame[y1:y2, x1:x2]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Pressione ESPACO para analisar", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Reconhecimento de Letras (Libras)", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:
        print("⏳ Aguarde 5 segundos...")
        cv2.putText(frame, "Aguardando 5 segundos...", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Reconhecimento de Letras (Libras)", frame)
        cv2.waitKey(1)
        time.sleep(5)

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi = frame[y1:y2, x1:x2]

        roi_resized = cv2.resize(roi, TAMANHO_IMAGEM)
        roi_normalized = roi_resized.astype('float32') / 255.0
        roi_reshaped = np.expand_dims(roi_normalized, axis=0)

        inicio = time.time()
        prediction = model.predict(roi_reshaped)
        fim = time.time()

        tempo_analise = fim - inicio
        predicted_class = class_names[np.argmax(prediction)]

        print(f"Letra identificada: {predicted_class}")
        print(f"Tempo de análise: {tempo_analise:.2f} segundos")

        cv2.putText(frame, f"Letra: {predicted_class}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Tempo: {tempo_analise:.2f}s", (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Reconhecimento de Letras (Libras)", frame)
        cv2.waitKey(2000)

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
