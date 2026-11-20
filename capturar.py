# capturar.py
# Toma fotos con la webcam y las guarda en ./extra/<clase>/ para reentrenar.
# Se guarda solo lo que está dentro del recuadro (lo mismo que ve camara.py).
#
# Uso: python capturar.py        (python capturar.py 1  para otra cámara)
#
# Teclas:
#   1-7      elegir clase
#   ESPACIO  tomar una foto
#   r        ráfaga ON/OFF (una foto cada 0.3 s; gira el objeto mientras tanto)
#   q / ESC  salir
import os
import sys
import time
from datetime import datetime

import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
EXTRA = os.path.join(BASE, "extra")

# tecla: (carpeta, texto en pantalla, color BGR del tacho)
CLASES = {
    "1": ("glass",           "Vidrio - VERDE",            (0, 170, 0)),
    "2": ("plastic",         "Plastico - BLANCO",         (255, 255, 255)),
    "3": ("metal",           "Metal - AMARILLO",          (0, 215, 255)),
    "4": ("peligroso",       "Peligroso - ROJO",          (0, 0, 220)),
    "5": ("paper_cardboard", "Papel/carton - AZUL",       (200, 90, 0)),
    "6": ("biological",      "Organico - MARRON",         (19, 69, 139)),
    "7": ("trash",           "No aprovechable - NEGRO",   (40, 40, 40)),
}
TAM_GUARDADO = 384     # px del lado de la foto guardada
INTERVALO_RAFAGA = 0.3  # segundos

for carpeta, _, _ in CLASES.values():
    os.makedirs(os.path.join(EXTRA, carpeta), exist_ok=True)


def contar(carpeta):
    return len(os.listdir(os.path.join(EXTRA, carpeta)))


def texto(img, msg, pos, escala=0.65, color=(255, 255, 255), grosor=2):
    cv2.putText(img, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor + 3, cv2.LINE_AA)
    cv2.putText(img, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def guardar(roi, carpeta):
    nombre = f"webcam_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
    foto = cv2.resize(roi, (TAM_GUARDADO, TAM_GUARDADO), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(EXTRA, carpeta, nombre), foto, [cv2.IMWRITE_JPEG_QUALITY, 92])


cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(cam)
if not cap.isOpened():
    sys.exit(f"No pude abrir la cámara {cam}")

actual = "1"
rafaga = False
ultimo = 0.0
flash = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    lado = int(min(h, w) * 0.7)
    x0, y0 = (w - lado) // 2, (h - lado) // 2
    roi = frame[y0:y0 + lado, x0:x0 + lado].copy()   # copia sin textos encima

    carpeta, etiqueta, color = CLASES[actual]

    if rafaga and time.time() - ultimo >= INTERVALO_RAFAGA:
        guardar(roi, carpeta)
        ultimo = time.time()
        flash = 2

    # Dibujo
    grosor = 10 if flash > 0 else 4
    flash = max(0, flash - 1)
    cv2.rectangle(frame, (x0, y0), (x0 + lado, y0 + lado), color, grosor)
    texto(frame, f"Clase: [{actual}] {etiqueta}  ({contar(carpeta)} fotos)", (10, 28), 0.7)
    if rafaga:
        texto(frame, "RAFAGA ON - gira el objeto", (10, 58), 0.7, (0, 0, 255))
    for i, (k, (c, et, _)) in enumerate(CLASES.items()):
        texto(frame, f"{k}: {et} ({contar(c)})", (10, h - 10 - (len(CLASES) - 1 - i) * 22), 0.5, grosor=1)
    texto(frame, "ESPACIO foto | r rafaga | q salir", (w - 330, h - 10), 0.5, grosor=1)

    cv2.imshow("Capturar fotos para entrenamiento", frame)
    tecla = cv2.waitKey(1) & 0xFF
    if tecla in (ord("q"), 27):
        break
    elif chr(tecla) in CLASES:
        actual = chr(tecla)
        rafaga = False
    elif tecla == ord(" "):
        guardar(roi, carpeta)
        flash = 3
    elif tecla == ord("r"):
        rafaga = not rafaga

cap.release()
cv2.destroyAllWindows()
print("\nFotos en ./extra:")
for c, _, _ in CLASES.values():
    print(f"  {c}: {contar(c)}")
