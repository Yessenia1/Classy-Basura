# trash.py
# Arma ./dataset_final juntando Kaggle + RealWaste + tus fotos propias (./extra)
import os, shutil
from PIL import Image

DEST = "./dataset_final"
EXTRA = "./extra"   # pon aquí tus fotos: ./extra/peligroso/foco1.jpg, etc.

CLASES_FINALES = ["peligroso", "biological", "glass", "metal", "paper_cardboard", "plastic", "trash"]
EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp")

FUENTES = [
    ("kaggle", "./Garbage_Dataset/standardized_256", {
        "battery": "peligroso",
        "biological": "biological",
        "glass": "glass",
        "metal": "metal",
        "paper": "paper_cardboard",
        "cardboard": "paper_cardboard",
        "plastic": "plastic",
        "trash": "trash",
        "shoes": "trash",
        "clothes": "trash",
    }),
    ("realwaste", "./realwaste_main/RealWaste", {
        "Cardboard": "paper_cardboard",
        "Paper": "paper_cardboard",
        "Food Organics": "biological",
        "Vegetation": "biological",
        "Glass": "glass",
        "Metal": "metal",
        "Plastic": "plastic",
        "Miscellaneous Trash": "trash",
        "Textile Trash": "trash",
    }),
    # TrashBox: https://github.com/nikhilvenkatkumsetty/TrashBox (solo carpetas útiles)
    ("trashbox", "./trashbox_main/TrashBox_train_dataset_subfolders", {
        # Tacho ROJO (peligrosos)
        "medical/medicines": "peligroso",
        "medical/syringe": "peligroso",
        "medical/gloves": "peligroso",
        "medical/masks": "peligroso",
        "metal/spray cans": "peligroso",          # aerosoles
        "e-waste/smartphones": "peligroso",       # RAEE
        "e-waste/electronic chips": "peligroso",  # RAEE
        # Refuerzo vidrio / plástico / metal
        "glass": "glass",
        "plastic/plastic bottles": "plastic",
        "plastic/plastic containers": "plastic",
        "plastic/plastic bags": "plastic",
        "metal/beverage cans": "metal",
        "metal/metal containers": "metal",
    }),
    # Recortes de TACO: envolturas brillantes y metales variados (los genera taco_datos.py)
    ("taco", "./taco_crops", {c: c for c in CLASES_FINALES}),
    # Recortes de Roboflow Universe (los genera roboflow_datos.py)
    ("roboflow", "./roboflow_crops", {c: c for c in CLASES_FINALES}),
    ("extra", EXTRA, {c: c for c in CLASES_FINALES}),
]

# Se rehace desde cero para no mezclar con versiones anteriores
if os.path.isdir(DEST):
    shutil.rmtree(DEST)
for c in CLASES_FINALES:
    os.makedirs(os.path.join(DEST, c), exist_ok=True)
    os.makedirs(os.path.join(EXTRA, c), exist_ok=True)


FIRMAS = (b"\xff\xd8", b"\x89PNG", b"BM", b"GIF8")   # formatos que lee TensorFlow


def imagen_valida(ruta):
    """Descarta archivos dañados o en formatos que TensorFlow no lee (ej. webp renombrado)."""
    try:
        with open(ruta, "rb") as f:
            if not f.read(4).startswith(FIRMAS):
                return False
        with Image.open(ruta) as im:
            im.load()
        return True
    except Exception:
        return False


descartadas = []


def copiar_dataset(path_origen, mapeo, prefijo):
    for clase_origen, clase_destino in mapeo.items():
        carpeta_origen = os.path.join(path_origen, clase_origen)
        if not os.path.isdir(carpeta_origen):
            print(f"No encontré carpeta: {carpeta_origen}")
            continue
        carpeta_destino = os.path.join(DEST, clase_destino)
        n = 0
        for archivo in os.listdir(carpeta_origen):
            if not archivo.lower().endswith(EXTENSIONES):
                continue
            if not imagen_valida(os.path.join(carpeta_origen, archivo)):
                descartadas.append(os.path.join(carpeta_origen, archivo))
                continue
            sub = clase_origen.replace(" ", "_").replace("/", "-")
            nombre = f"{prefijo}_{sub}_{archivo.replace(' ', '_')}"
            shutil.copy(os.path.join(carpeta_origen, archivo), os.path.join(carpeta_destino, nombre))
            n += 1
        print(f"  {prefijo}/{clase_origen} -> {clase_destino}: {n}")


for prefijo, ruta, mapeo in FUENTES:
    print(f"Copiando {prefijo}...")
    copiar_dataset(ruta, mapeo, prefijo)

if descartadas:
    print(f"\nDescartadas por estar dañadas o en formato no soportado: {len(descartadas)}")
    for r in descartadas[:10]:
        print("  ", r)

print("\nTotal por clase:")
for c in CLASES_FINALES:
    print(f"  {c}: {len(os.listdir(os.path.join(DEST, c)))} imágenes")
