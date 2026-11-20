import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# ---------- 1. Cargar datasets ----------
IMG_SIZE = (224, 224)   # tamaño estándar para MobileNetV2
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "./dataset_split/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "./dataset_split/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

CLASS_NAMES = train_ds.class_names
print("Clases (en orden que usará el modelo):", CLASS_NAMES)

# ---------- 2. Class weights (por el desbalance) ----------
# Se cuentan las imágenes de train directamente de las carpetas
import os, json
conteo = {c: len(os.listdir(os.path.join("./dataset_split/train", c))) for c in CLASS_NAMES}
print("Imágenes de train por clase:", conteo)
total = sum(conteo.values())
n_clases = len(CLASS_NAMES)

class_weights = {}
for i, nombre in enumerate(CLASS_NAMES):
    class_weights[i] = total / (n_clases * conteo[nombre])

print("Class weights:", class_weights)

# Se guardan para fase 2, predecir.py y camara.py
with open("clases.json", "w", encoding="utf-8") as f:
    json.dump({"clases": CLASS_NAMES, "class_weights": class_weights}, f, indent=2)

# ---------- 3. Optimizar pipeline ----------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---------- 4. Data augmentation ----------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.15),
])

# ---------- 5. Modelo base (transfer learning) ----------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # congelamos capas base por ahora

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(n_clases, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------- 6. Entrenar (fase 1: solo la cabeza) ----------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights
)

model.save("modelo_fase1.keras")
print("Fase 1 completada y guardada")