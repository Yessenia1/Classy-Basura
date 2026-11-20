# camara.py
# Clasifica en vivo lo que pongas dentro del recuadro de la cámara.
# Uso: python camara.py        (q o ESC para salir)
#      python camara.py 1      (para usar otra cámara)
import sys
import unicodedata
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from config import INFO, UMBRAL, cargar_clases

CLASES = cargar_clases()
model = tf.keras.models.load_model("modelo_mejor.keras")

N_SUAVIZADO = 8   # promedia los últimos N cuadros para que el resultado no "salte"


def sin_tildes(texto):
    # cv2.putText no dibuja tildes ni ñ
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


def texto(img, msg, pos, escala=0.7, color=(255, 255, 255), grosor=2):
    cv2.putText(img, sin_tildes(msg), pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor + 3, cv2.LINE_AA)
    cv2.putText(img, sin_tildes(msg), pos, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(cam)
if not cap.isOpened():
    sys.exit(f"No pude abrir la cámara {cam}")

historial = deque(maxlen=N_SUAVIZADO)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)   # efecto espejo, más natural
    h, w = frame.shape[:2]

    # Recuadro central cuadrado: ahí se pone el objeto
    lado = int(min(h, w) * 0.7)
    x0, y0 = (w - lado) // 2, (h - lado) // 2
    roi = frame[y0:y0 + lado, x0:x0 + lado]

    x = cv2.cvtColor(cv2.resize(roi, (224, 224)), cv2.COLOR_BGR2RGB).astype("float32")
    probs = model(x[None], training=False).numpy()[0]
    historial.append(probs)
    media = np.mean(historial, axis=0)

    mejor = int(np.argmax(media))
    conf = float(media[mejor])

    if conf < UMBRAL:
        color_bgr = (160, 160, 160)
        linea1 = "No estoy seguro..."
        linea2 = "Acerca el objeto al recuadro"
    else:
        nombre, tacho, ejemplos, color_bgr = INFO[CLASES[mejor]]
        linea1 = f"{nombre} ({conf*100:.0f}%)"
        linea2 = f"Tacho {tacho}"

    # Recuadro del color del tacho
    cv2.rectangle(frame, (x0, y0), (x0 + lado, y0 + lado), color_bgr, 4)
    # Barra inferior con el resultado
    cv2.rectangle(frame, (0, h - 80), (w, h), color_bgr, -1)
    texto(frame, linea1, (15, h - 48), 0.9)
    texto(frame, linea2, (15, h - 15), 0.9)

    # Top 3 arriba a la izquierda
    for i, k in enumerate(np.argsort(media)[::-1][:3]):
        texto(frame, f"{CLASES[k]}: {media[k]*100:.0f}%", (10, 25 + i * 25), 0.6, grosor=1)

    cv2.imshow("Clasificador de residuos", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break

cap.release()
cv2.destroyAllWindows()
