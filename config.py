# config.py
# Datos compartidos por predecir.py y camara.py
import json
import os

ARCHIVO_CLASES = "clases.json"   # lo genera entrenar.py
UMBRAL = 0.60                    # debajo de esto, el modelo "no está seguro"

# Tacho para envolturas brillantes de snacks/galletas (plástico metalizado multicapa).
# La guía peruana NTP 900.058 suele mandarlas al NEGRO (no aprovechable); cambia a "trash" si prefieres eso.
# Lo usan taco_datos.py y roboflow_datos.py al armar los datos.
ENVOLTURAS = "plastic"

# Orden de clases del modelo antiguo (antes de clases.json)
CLASES_ANTIGUAS = ["battery", "biological", "glass", "metal", "paper_cardboard", "plastic", "trash"]

# clase: (nombre, color del tacho, ejemplos, color BGR para OpenCV)
INFO = {
    "peligroso":       ("Residuo peligroso",       "ROJO",     "pilas, baterías, focos, medicamentos, aerosoles, jeringas", (0, 0, 220)),
    "biological":      ("Residuo orgánico",        "MARRÓN",   "restos de comida, cáscaras, hojas",                         (19, 69, 139)),
    "glass":           ("Vidrio",                  "VERDE",    "botellas y frascos",                                        (0, 170, 0)),
    "metal":           ("Metal",                   "AMARILLO", "latas, tapas, chatarra",                                    (0, 215, 255)),
    "paper_cardboard": ("Papel / cartón",          "AZUL",     "hojas, periódicos, cajas",                                  (200, 90, 0)),
    "plastic":         ("Plástico",                "BLANCO",   "botellas PET, envases, bolsas",                             (255, 255, 255)),
    "trash":           ("Residuo no aprovechable", "NEGRO",    "papel higiénico, ropa, calzado, tecnopor sucio",            (40, 40, 40)),
}
INFO["battery"] = INFO["peligroso"]   # compatibilidad con el modelo antiguo


def cargar_clases():
    """Lee el orden de clases que guardó entrenar.py."""
    if os.path.exists(ARCHIVO_CLASES):
        with open(ARCHIVO_CLASES, encoding="utf-8") as f:
            return json.load(f)["clases"]
    return CLASES_ANTIGUAS
