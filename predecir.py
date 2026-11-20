# predecir.py
# Uso: python predecir.py foto1.jpg foto2.jpg ...
import sys
import numpy as np
import tensorflow as tf
from config import INFO, UMBRAL, cargar_clases

CLASES = cargar_clases()
model = tf.keras.models.load_model("modelo_mejor.keras")

for ruta in sys.argv[1:]:
    img = tf.keras.utils.load_img(ruta, target_size=(224, 224))
    x = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)
    probs = model.predict(x, verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]

    print(f"\n {ruta}")
    for k in top:
        print(f"   {CLASES[k]:16s} {probs[k]*100:5.1f}%")

    mejor = top[0]
    if probs[mejor] < UMBRAL:
        print("   No estoy seguro, acomoda mejor el objeto y vuelve a intentar")
    else:
        nombre, color, ejemplos, _ = INFO[CLASES[mejor]]
        print(f"   ➜ {nombre}  →  tacho {color}  ({ejemplos})")
