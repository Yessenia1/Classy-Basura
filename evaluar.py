import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "./dataset_split/test",
    image_size=IMG_SIZE,
    batch_size=32,
    label_mode="categorical",
    shuffle=False          # importante: mantiene el orden para comparar
)
CLASES = test_ds.class_names

model = tf.keras.models.load_model("modelo_mejor.keras")

loss, acc = model.evaluate(test_ds)
print(f"\nAccuracy en TEST: {acc:.4f}")

# Predicciones vs. etiquetas reales
y_true = np.concatenate([np.argmax(y, axis=1) for _, y in test_ds])
y_pred = np.argmax(model.predict(test_ds), axis=1)

print("\n", classification_report(y_true, y_pred, target_names=CLASES, digits=3))

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 7))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(CLASES)))
ax.set_yticks(range(len(CLASES)))
ax.set_xticklabels(CLASES, rotation=45, ha="right")
ax.set_yticklabels(CLASES)
for i in range(len(CLASES)):
    for j in range(len(CLASES)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
plt.tight_layout()
plt.savefig("matriz_confusion.png")
print("Matriz guardada en matriz_confusion.png")