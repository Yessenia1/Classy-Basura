# exportar_pytorch.py
# Convierte modelo_mejor.keras (TensorFlow) a modelo.pt (PyTorch / TorchScript)
# para ejecutarlo en una Raspberry Pi 4. No reentrena: copia los pesos tal cual.
#
# Resultado en ./entrega_rpi/: modelo.pt, clases.json (+ camara_rpi.py y LEEME.txt)
import json
import os

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

SALIDA = "entrega_rpi"
EPS = 1e-3   # epsilon de BatchNorm en Keras


class ConvBN(nn.Module):
    """Conv (normal o depthwise) + BatchNorm + ReLU6 opcional, con el mismo relleno que Keras."""

    def __init__(self, c_in, c_out, k, stride=1, depthwise=False, relu=True, pad_tf=False):
        super().__init__()
        # pad_tf: relleno asimétrico (0 arriba/izq, 1 abajo/der) como 'same'/ZeroPadding2D de Keras con stride 2
        self.pad_tf = pad_tf
        padding = 0 if (pad_tf or k == 1) else k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, stride, padding, groups=c_in if depthwise else 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out, eps=EPS)
        self.relu = relu

    def forward(self, x):
        if self.pad_tf:
            x = F.pad(x, (0, 1, 0, 1))
        x = self.bn(self.conv(x))
        return F.relu6(x) if self.relu else x


class Bloque(nn.Module):
    def __init__(self, c_in, c_out, expandir, stride, residual):
        super().__init__()
        c_mid = c_in * expandir
        self.expand = ConvBN(c_in, c_mid, 1) if expandir != 1 else None
        self.dw = ConvBN(c_mid, c_mid, 3, stride, depthwise=True, pad_tf=(stride == 2))
        self.project = ConvBN(c_mid, c_out, 1, relu=False)
        self.residual = residual

    def forward(self, x):
        y = self.expand(x) if self.expand is not None else x
        y = self.project(self.dw(y))
        return x + y if self.residual else y


class ClasificadorResiduos(nn.Module):
    """MobileNetV2 + GlobalAveragePooling + Dense(softmax), igual que entrenar.py.

    Entrada: tensor (N, 224, 224, 3) float con valores RGB de 0 a 255.
    Salida:  probabilidades (N, 7).
    """

    def __init__(self, n_clases, config):
        super().__init__()
        self.stem = ConvBN(3, 32, 3, 2, pad_tf=True)
        self.bloques = nn.Sequential(*[Bloque(*c) for c in config])
        self.head = ConvBN(320, 1280, 1)
        self.fc = nn.Linear(1280, n_clases)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) / 127.5 - 1.0     # preprocess_input de MobileNetV2
        x = self.head(self.bloques(self.stem(x)))
        x = x.mean(dim=(2, 3))
        return torch.softmax(self.fc(x), dim=1)


# ---------- 1. Leer el modelo de Keras ----------
keras_model = tf.keras.models.load_model("modelo_mejor.keras")
base = keras_model.get_layer("mobilenetv2_1.00_224")
capa = {l.name: l for l in base.layers}
with open("clases.json", encoding="utf-8") as f:
    CLASES = json.load(f)["clases"]

# Configuración de bloques sacada del propio modelo (canales, expansión, stride, residual)
config = []
c_in = 16
config.append((32, 16, 1, 1, False))   # expanded_conv (bloque 0, sin expansión)
for i in range(1, 17):
    exp = capa[f"block_{i}_expand"]
    proj = capa[f"block_{i}_project"]
    c_mid = exp.filters
    c_out = proj.filters
    stride = capa[f"block_{i}_depthwise"].strides[0]
    config.append((c_in, c_out, c_mid // c_in, stride, f"block_{i}_add" in capa))
    c_in = c_out

torch_model = ClasificadorResiduos(len(CLASES), config).eval()


# ---------- 2. Copiar pesos ----------
def copiar_conv(destino, nombre_conv, nombre_bn, depthwise=False):
    w = capa[nombre_conv].get_weights()[0]
    w = w.transpose(2, 3, 0, 1) if depthwise else w.transpose(3, 2, 0, 1)
    destino.conv.weight.data = torch.from_numpy(np.ascontiguousarray(w))
    gamma, beta, media, var = capa[nombre_bn].get_weights()
    destino.bn.weight.data = torch.from_numpy(gamma)
    destino.bn.bias.data = torch.from_numpy(beta)
    destino.bn.running_mean.data = torch.from_numpy(media)
    destino.bn.running_var.data = torch.from_numpy(var)


copiar_conv(torch_model.stem, "Conv1", "bn_Conv1")
b0 = torch_model.bloques[0]
copiar_conv(b0.dw, "expanded_conv_depthwise", "expanded_conv_depthwise_BN", depthwise=True)
copiar_conv(b0.project, "expanded_conv_project", "expanded_conv_project_BN")
for i in range(1, 17):
    b = torch_model.bloques[i]
    copiar_conv(b.expand, f"block_{i}_expand", f"block_{i}_expand_BN")
    copiar_conv(b.dw, f"block_{i}_depthwise", f"block_{i}_depthwise_BN", depthwise=True)
    copiar_conv(b.project, f"block_{i}_project", f"block_{i}_project_BN")
copiar_conv(torch_model.head, "Conv_1", "Conv_1_bn")
w, bias = keras_model.get_layer("dense").get_weights()
torch_model.fc.weight.data = torch.from_numpy(np.ascontiguousarray(w.T))
torch_model.fc.bias.data = torch.from_numpy(bias)


# ---------- 3. Verificar contra Keras ----------
test_ds = tf.keras.utils.image_dataset_from_directory(
    "./dataset_split/test", image_size=(224, 224), batch_size=64, shuffle=True, seed=7)
imgs, _ = next(iter(test_ds))
p_keras = keras_model.predict(imgs, verbose=0)
with torch.no_grad():
    p_torch = torch_model(torch.from_numpy(imgs.numpy())).numpy()
iguales = int((p_keras.argmax(1) == p_torch.argmax(1)).sum())
dif = float(np.abs(p_keras - p_torch).max())
print(f"PyTorch vs Keras: {iguales}/{len(imgs)} predicciones iguales, diferencia máx. {dif:.6f}")
assert iguales == len(imgs) and dif < 1e-3, "La conversión no coincide con Keras"


# ---------- 4. Guardar ----------
os.makedirs(SALIDA, exist_ok=True)
ejemplo = torch.zeros(1, 224, 224, 3)
with torch.no_grad():
    script = torch.jit.trace(torch_model, ejemplo)
ruta = os.path.join(SALIDA, "modelo.pt")
script.save(ruta, _extra_files={"clases.json": json.dumps(CLASES)})
with open(os.path.join(SALIDA, "clases.json"), "w", encoding="utf-8") as f:
    json.dump(CLASES, f)
print(f"Guardado {ruta} ({os.path.getsize(ruta)/1e6:.1f} MB)")

# Comprobación final: se carga sin necesitar este código ni TensorFlow
cargado = torch.jit.load(ruta)
with torch.no_grad():
    p2 = cargado(torch.from_numpy(imgs.numpy()[:8])).numpy()
print("Recarga OK:", bool(np.allclose(p2, p_torch[:8], atol=1e-5)))
