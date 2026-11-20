# roboflow_datos.py
# Descarga datasets de detección de Roboflow Universe y recorta cada objeto
# marcado como una foto individual en ./roboflow_crops/<clase>/
#
# Uso (PowerShell):
#   $env:ROBOFLOW_API_KEY = "tu_clave"     # roboflow.com > Settings > API Keys
#   python roboflow_datos.py
#
# Luego: python trash.py  y  python split_dataset.py
import io
import json
import os
import re
import shutil
import urllib.request
import zipfile

from PIL import Image

API_KEY = os.environ.get("ROBOFLOW_API_KEY")
SALIDA = "./roboflow_crops"
DESCARGAS = "./roboflow_descargas"
MAX_POR_CLASE = {"peligroso": 500}   # tope por clase en cada dataset
MAX_DEFECTO = 800
MAX_POR_FOTO = 4        # recortes por foto, para tener más variedad
MIN_LADO = 48           # px; recortes más chicos se descartan
MARGEN = 0.10           # agrega 10 % alrededor del recuadro

from config import ENVOLTURAS   # tacho para envolturas brillantes de snacks

# Reglas por nombre de clase (se revisan en orden; None = ignorar)
REGLAS = [
    (r"non.?plastic|^broken$|^crisp$", None),
    (r"glass|vidrio|ClearBottle|green ?bottle", "glass"),
    (r"wrapper|crisp.?packet|snack|chips?\b|envoltura|packet", ENVOLTURAS),
    (r"plastic|\bpet\b|plastico", "plastic"),
    (r"\bcans?\b|\blatas?\b|metal|alumin", "metal"),
    (r"bulb|lamp|foco|bombilla|battery|bateria|pila", "peligroso"),
]

# workspace/proyecto, y clase para nombres genéricos (ej. "bottle")
DATASETS = [
    ("klares-workspace/bottle-detection-cer5j", None),
    ("circularity-7fb36/glass-bottles-cnquk", ("bottle", "glass")),
    ("bote/plastic-bottles-9gobq", ("bottle", "plastic")),
    ("manual-datasets/plastic-bottles-100", ("bottle", "plastic")),
    ("dems-trash/lightbulb-21zws", ("light", "peligroso")),
    ("tanmay-singh-g3brk/bulb-sqljc", ("light", "peligroso")),
    ("new-workspace-wfilj/light_bulb_open_image", ("light", "peligroso")),
    # Bolsas de snacks (papas fritas, galletas): plástico metalizado que se confunde con metal.
    # Sus clases suelen ser marcas, por eso "." manda cualquier nombre a ENVOLTURAS.
    ("meanzaza/chips-slo4r", (".", ENVOLTURAS)),
    ("chananya/snackbag", (".", ENVOLTURAS)),
    ("test-5hmcs/crisps", ("seabrook", ENVOLTURAS)),
]


def clase_destino(nombre, generico):
    for patron, clase in REGLAS:
        if re.search(patron, nombre, re.I):
            return clase
    if generico and re.search(generico[0], nombre, re.I):
        return generico[1]
    return None


def api(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def descargar(slug):
    """Baja la última versión en formato YOLOv8 y devuelve la carpeta."""
    destino = os.path.join(DESCARGAS, slug.replace("/", "__"))
    if os.path.isdir(destino):
        return destino
    info = api(f"https://api.roboflow.com/{slug}?api_key={API_KEY}")
    versiones = info.get("versions") or []
    if not versiones:
        raise RuntimeError("no tiene versiones publicadas")
    ultima = versiones[-1]["id"].split("/")[-1]
    exp = api(f"https://api.roboflow.com/{slug}/{ultima}/yolov8?api_key={API_KEY}")
    link = exp["export"]["link"]
    os.makedirs(DESCARGAS, exist_ok=True)
    zip_path = destino + ".zip"
    with urllib.request.urlopen(link, timeout=600) as r, open(zip_path + ".part", "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        bajado = 0
        while True:
            bloque = r.read(1 << 20)
            if not bloque:
                break
            f.write(bloque)
            bajado += len(bloque)
            if total:
                print(f"\r   descargando {bajado/1e6:6.1f} / {total/1e6:.1f} MB ({bajado*100/total:3.0f}%)", end="", flush=True)
            else:
                print(f"\r   descargando {bajado/1e6:6.1f} MB", end="", flush=True)
    print()
    os.replace(zip_path + ".part", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(destino)
    os.remove(zip_path)
    return destino


def leer_nombres(carpeta):
    with open(os.path.join(carpeta, "data.yaml"), encoding="utf-8") as f:
        texto = f.read()
    m = re.search(r"names:\s*\[(.*?)\]", texto, re.S)
    if m:
        return [n.strip().strip("'\"") for n in m.group(1).split(",")]
    # formato en lista:  - nombre   o   0: nombre
    bloque = texto.split("names:", 1)[1]
    return re.findall(r"^\s*(?:-|\d+:)\s*['\"]?(.+?)['\"]?\s*$", bloque, re.M)


def recortar(carpeta, slug, generico):
    nombres = leer_nombres(carpeta)
    mapa = {i: clase_destino(n, generico) for i, n in enumerate(nombres)}
    print("   clases:", {nombres[i]: mapa[i] for i in mapa})
    prefijo = slug.split("/")[1]
    cuenta = {}
    vistas = set()   # Roboflow exporta varias copias aumentadas de la misma foto
    for raiz, _, archivos in os.walk(carpeta):
        if not raiz.replace("\\", "/").endswith("/labels"):
            continue
        dir_img = raiz[:-len("labels")] + "images"
        for arch in sorted(archivos):
            base = os.path.splitext(arch)[0]
            original = re.sub(r"_(jpe?g|png)\.rf\..*$", "", base, flags=re.I)
            if original in vistas:
                continue
            vistas.add(original)
            img_path = next((os.path.join(dir_img, base + e) for e in (".jpg", ".jpeg", ".png")
                             if os.path.exists(os.path.join(dir_img, base + e))), None)
            if not img_path:
                continue
            with open(os.path.join(raiz, arch)) as f:
                lineas = [l.split() for l in f if l.strip()]
            img = None
            en_foto = 0
            for j, partes in enumerate(lineas):
                clase = mapa.get(int(partes[0]))
                if clase is None:
                    continue
                if en_foto >= MAX_POR_FOTO or cuenta.get(clase, 0) >= MAX_POR_CLASE.get(clase, MAX_DEFECTO):
                    continue
                valores = [float(v) for v in partes[1:]]
                if len(valores) == 4:                       # recuadro: cx cy w h
                    cx, cy, w, h = valores
                elif len(valores) >= 6 and len(valores) % 2 == 0:   # polígono: x1 y1 x2 y2 ...
                    xs, ys = valores[0::2], valores[1::2]
                    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                    w, h = max(xs) - min(xs), max(ys) - min(ys)
                else:
                    continue
                if img is None:
                    img = Image.open(img_path).convert("RGB")
                W, H = img.size
                w, h = w * (1 + 2 * MARGEN), h * (1 + 2 * MARGEN)
                x0, y0 = max(0, (cx - w / 2) * W), max(0, (cy - h / 2) * H)
                x1, y1 = min(W, (cx + w / 2) * W), min(H, (cy + h / 2) * H)
                if min(x1 - x0, y1 - y0) < MIN_LADO:
                    continue
                if max(x1 - x0, y1 - y0) > 4 * min(x1 - x0, y1 - y0):   # tiras muy delgadas (objeto cortado)
                    continue
                os.makedirs(os.path.join(SALIDA, clase), exist_ok=True)
                img.crop((x0, y0, x1, y1)).save(
                    os.path.join(SALIDA, clase, f"{prefijo}_{original[:60]}_{j}.jpg"), quality=92)
                cuenta[clase] = cuenta.get(clase, 0) + 1
                en_foto += 1
    return cuenta


if __name__ == "__main__":
    # Los recortes se rehacen desde cero (las descargas se reutilizan)
    if os.path.isdir(SALIDA):
        shutil.rmtree(SALIDA)
    for slug, generico in DATASETS:
        print(f"\n{slug}")
        ya_bajado = os.path.isdir(os.path.join(DESCARGAS, slug.replace("/", "__")))
        if not ya_bajado and not API_KEY:
            print("   OMITIDO: falta la clave ($env:ROBOFLOW_API_KEY = \"tu_clave\")")
            continue
        try:
            carpeta = descargar(slug)
            print(f"   recortes: {recortar(carpeta, slug, generico)}")
        except Exception as e:
            print(f"   OMITIDO: {e}")

    print("\nRecortes en", SALIDA)
    for c in sorted(os.listdir(SALIDA)) if os.path.isdir(SALIDA) else []:
        print(f"  {c}: {len(os.listdir(os.path.join(SALIDA, c)))}")
