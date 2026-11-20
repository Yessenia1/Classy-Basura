import tensorflow as tf

# ---------- 1. Recargar datasets (igual que antes) ----------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "./dataset_split/train", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "./dataset_split/val", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical"
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Pesos calculados en entrenar.py (fase 1)
import json
with open("clases.json", encoding="utf-8") as f:
    class_weights = {int(k): v for k, v in json.load(f)["class_weights"].items()}

# ---------- 2. Cargar el modelo de la fase 1 ----------
model = tf.keras.models.load_model("modelo_fase1.keras")

# ---------- 3. Descongelar las últimas capas de MobileNetV2 ----------
base_model = model.get_layer("mobilenetv2_1.00_224")
base_model.trainable = True

# Congelamos todo excepto las últimas ~40 capas (las más "específicas")
fine_tune_at = len(base_model.layers) - 40
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# ---------- 4. Recompilar con LR mucho más bajo ----------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------- 5. Callbacks: guardar solo el mejor modelo ----------
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "modelo_mejor.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

# ---------- 6. Entrenar fase 2 ----------
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop]
)

print("Fine-tuning completado. Mejor modelo guardado en modelo_mejor.keras")