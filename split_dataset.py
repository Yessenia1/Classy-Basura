# split_dataset.py
import os, shutil
import splitfolders

# Borra el split anterior para que no queden clases viejas (ej. "battery")
if os.path.isdir("./dataset_split"):
    shutil.rmtree("./dataset_split")

splitfolders.ratio(
    "./dataset_final",
    output="./dataset_split",
    seed=42,
    ratio=(0.7, 0.15, 0.15)
)

print("Split completado")
