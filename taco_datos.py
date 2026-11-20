# taco_datos.py
# Descarga objetos del dataset TACO (basura real fotografiada en la calle, licencia MIT)
# y recorta cada objeto marcado como foto individual en ./taco_crops/<clase>/
#   https://github.com/pedropro/TACO   http://tacodataset.org
#
# Uso: python taco_datos.py      (no necesita cuenta ni clave)
# Luego: python trash.py  y  python split_dataset.py
import io
import json
import os
import shutil
import threading
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps

SALIDA = "./taco_crops"
CACHE = "./taco_descargas"          # anotaciones + fotos originales (se reutilizan)
MIN_LADO = 64                       # px; objetos más chicos se descartan
MARGEN = 0.15                       # 15 % alrededor del objeto
MAX_LADO_GUARDADO = 512             # los recortes grandes se achican a esto

from config import ENVOLTURAS   # "plastic" o "trash": se cambia en config.py

# Categoría TACO -> (clase del proyecto, máximo de recortes)
MAPEO = {
    # Envolturas: lo que hoy se confunde con metal
    "Crisp packet":           (ENVOLTURAS, 600),
    "Other plastic wrapper":  (ENVOLTURAS, 600),
    # Metales poco representados en tus datos
    "Aluminium foil":         ("metal", 400),
    "Metal bottle cap":       ("metal", 400),
    "Pop tab":                ("metal", 400),
    "Scrap metal":            ("metal", 400),
    "Food Can":               ("metal", 400),
    "Metal lid":              ("metal", 400),
    "Drink can":              ("metal", 300),
    # Peligrosos
    "Aerosol":                ("peligroso", 100),
    "Battery":                ("peligroso", 100),
    "Aluminium blister pack": ("peligroso", 100),   # blíster de pastillas
}

_lock_zip = threading.Lock()
URL_ANOT = "https://raw.githubusercontent.com/pedropro/TACO/master/data/"
ARCHIVOS_ANOT = ["annotations.json", "annotations_unofficial.json"]


def bajar(url, timeout=60):
    for intento in range(4):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:
            if intento == 3:
                raise


def cargar_anotaciones():
    os.makedirs(CACHE, exist_ok=True)
    datos = []
    for nombre in ARCHIVOS_ANOT:
        ruta = os.path.join(CACHE, nombre)
        if not os.path.exists(ruta):
            with open(ruta, "wb") as f:
                f.write(bajar(URL_ANOT + nombre))
        with open(ruta, encoding="utf-8") as f:
            datos.append((nombre.split(".")[0], json.load(f)))
    return datos


# Las fotos se leen de:
#  1) TACO.zip oficial de Zenodo (conjunto principal, 1500 fotos, CC BY 4.0):
#     https://zenodo.org/records/3587843  -> guardar como ./taco_descargas/TACO.zip
#  2) fotos sueltas ya bajadas antes en ./taco_descargas/fotos/
# (el servidor img.tacodataset.org bloquea descargas masivas, así que no se usa)
ZIP_OFICIAL = os.path.join(CACHE, "TACO.zip")
_zip, _indice_zip = None, {}


def abrir_zip():
    global _zip
    if _zip is None and os.path.exists(ZIP_OFICIAL):
        _zip = zipfile.ZipFile(ZIP_OFICIAL)
        for n in _zip.namelist():
            partes = n.replace("\\", "/").split("/")
            if len(partes) >= 2 and partes[-1].lower().endswith((".jpg", ".jpeg", ".png")):
                _indice_zip["/".join(partes[-2:]).lower()] = n   # "batch_1/000006.jpg"
    return _zip


def foto_local(conjunto, img):
    """Devuelve un archivo (ruta o BytesIO) con la foto, o None si no está disponible."""
    ruta = os.path.join(CACHE, "fotos", conjunto, img["file_name"].replace("/", "_"))
    if os.path.exists(ruta):
        return ruta
    if conjunto == "annotations" and abrir_zip():
        nombre = _indice_zip.get(img["file_name"].lower())
        if nombre:
            with _lock_zip:
                return io.BytesIO(_zip.read(nombre))
    return None


def main():
    if os.path.isdir(SALIDA):
        shutil.rmtree(SALIDA)

    # 1. Elegir objetos de las categorías que interesan (respetando los topes)
    pendientes = {}          # (conjunto, image_id) -> [(anotación, clase, categoría)]
    imagenes = {}
    cuenta_cat = {}
    for conjunto, d in cargar_anotaciones():
        cats = {c["id"]: c["name"] for c in d["categories"]}
        imgs = {i["id"]: i for i in d["images"]}
        for a in d["annotations"]:
            cat = cats[a["category_id"]]
            if cat not in MAPEO:
                continue
            clase, tope = MAPEO[cat]
            if cuenta_cat.get(cat, 0) >= tope:
                continue
            w, h = a["bbox"][2], a["bbox"][3]
            if min(w, h) < MIN_LADO:
                continue
            cuenta_cat[cat] = cuenta_cat.get(cat, 0) + 1
            clave = (conjunto, a["image_id"])
            pendientes.setdefault(clave, []).append((a, clase, cat))
            imagenes[clave] = imgs[a["image_id"]]
    if not os.path.exists(ZIP_OFICIAL):
        print(f"AVISO: falta {ZIP_OFICIAL} (https://zenodo.org/records/3587843); solo se usan fotos ya guardadas.")
    print(f"Fotos con objetos útiles: {len(pendientes)}  (objetos: {sum(cuenta_cat.values())})", flush=True)

    # 2. Descargar fotos y recortar
    def procesar(clave):
        conjunto, _ = clave
        img_info = imagenes[clave]
        ruta = foto_local(conjunto, img_info)
        if not ruta:
            return []
        try:
            original = Image.open(ruta)
            original.load()
            img = ImageOps.exif_transpose(original).convert("RGB")
        except Exception:
            return []
        # TACO anota sobre la foto ya girada según EXIF; si el tamaño no cuadra, se usa sin girar
        if img.size != (img_info["width"], img_info["height"]):
            img = original.convert("RGB")
        if img.size != (img_info["width"], img_info["height"]):
            return []
        W, H = img.size
        hechos = []
        for a, clase, cat in pendientes[clave]:
            x, y, w, h = a["bbox"]
            mx, my = w * MARGEN, h * MARGEN
            caja = (max(0, x - mx), max(0, y - my), min(W, x + w + mx), min(H, y + h + my))
            if min(caja[2] - caja[0], caja[3] - caja[1]) < MIN_LADO:
                continue
            rec = img.crop(caja)
            rec.thumbnail((MAX_LADO_GUARDADO, MAX_LADO_GUARDADO))
            carpeta = os.path.join(SALIDA, clase)
            os.makedirs(carpeta, exist_ok=True)
            nombre = f"taco_{cat.replace(' ', '-')}_{conjunto}_{a['id']}.jpg"
            rec.save(os.path.join(carpeta, nombre), quality=92)
            hechos.append(cat)
        return hechos

    resumen, fallidas = {}, 0
    with ThreadPoolExecutor(12) as ex:
        for i, hechos in enumerate(ex.map(procesar, list(pendientes)), 1):
            if not hechos:
                fallidas += 1
            for cat in hechos:
                resumen[cat] = resumen.get(cat, 0) + 1
            if i % 200 == 0:
                print(f"  {i}/{len(pendientes)} fotos", flush=True)

    print("\nRecortes por categoría TACO:")
    for cat, n in sorted(resumen.items(), key=lambda t: -t[1]):
        print(f"  {cat:24s} -> {MAPEO[cat][0]:10s} {n}")
    if fallidas:
        print(f"Fotos sin recortes (no descargaron o sin objetos válidos): {fallidas}")
    print("\nTotal en", SALIDA)
    for c in sorted(os.listdir(SALIDA)):
        print(f"  {c}: {len(os.listdir(os.path.join(SALIDA, c)))}")


if __name__ == "__main__":
    main()
