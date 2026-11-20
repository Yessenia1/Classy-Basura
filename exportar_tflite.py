# exportar_tflite.py
# Convierte modelo_mejor.keras a TFLite para la app web del celular (carpeta ./app_web)
import json
import os

import numpy as np
import tensorflow as tf

SALIDA = "app_web"
os.makedirs(SALIDA, exist_ok=True)

model = tf.keras.models.load_model("modelo_mejor.keras")

# Modelo limpio para inferencia: se salta la capa de data augmentation
# (misma arquitectura que entrenar.py, reutilizando las capas ya entrenadas)
inputs = tf.keras.Input(shape=(224, 224, 3), name="imagen")
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = model.get_layer("mobilenetv2_1.00_224")(x, training=False)
x = model.get_layer("global_average_pooling2d")(x)
x = model.get_layer("dense")(x)
limpio = tf.keras.Model(inputs, x)

conv = tf.lite.TFLiteConverter.from_keras_model(limpio)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]   # pesa la mitad, casi sin perder precisión
tflite = conv.convert()

ruta = os.path.join(SALIDA, "modelo.tflite")
with open(ruta, "wb") as f:
    f.write(tflite)
print(f"Guardado {ruta} ({len(tflite)/1e6:.1f} MB)")

with open("clases.json", encoding="utf-8") as f:
    clases = json.load(f)["clases"]
with open(os.path.join(SALIDA, "clases.json"), "w", encoding="utf-8") as f:
    json.dump(clases, f)

# Verificación: TFLite debe dar lo mismo que Keras en imágenes de test
interp = tf.lite.Interpreter(model_path=ruta)
interp.allocate_tensors()
ent, sal = interp.get_input_details()[0], interp.get_output_details()[0]

test_ds = tf.keras.utils.image_dataset_from_directory(
    "./dataset_split/test", image_size=(224, 224), batch_size=32, shuffle=True, seed=1)
imgs, _ = next(iter(test_ds))
p_keras = model.predict(imgs, verbose=0)
coinciden, dif = 0, 0.0
for i in range(len(imgs)):
    interp.set_tensor(ent["index"], imgs[i:i + 1].numpy().astype(np.float32))
    interp.invoke()
    p = interp.get_tensor(sal["index"])[0]
    coinciden += int(np.argmax(p) == np.argmax(p_keras[i]))
    dif = max(dif, float(np.abs(p - p_keras[i]).max()))
print(f"TFLite vs Keras: {coinciden}/{len(imgs)} predicciones iguales, diferencia máx. {dif:.4f}")
