# inferencia_onnx.py
import os, time, psutil, gc
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = "/home/pi/modelos/best_quantized.onnx"
IMAGE_PATH = "/home/pi/imagens/teste.jpg"
INPUT_SIZE = (320, 320)   # ajuste conforme seu modelo
CONF_THRESHOLD = 0.3

def get_ram_mb():
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024**2)

def preprocess_image(img_path, size):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    # formato NCHW
    arr = arr.transpose(2,0,1)[None, ...]
    return arr

def run_inference():
    print("Carregando modelo...")
    t0 = time.time()
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    t1 = time.time()
    print(f"Modelo carregado em {t1-t0:.3f}s")

    input_name = session.get_inputs()[0].name
    x = preprocess_image(IMAGE_PATH, INPUT_SIZE)

    ram_before = get_ram_mb()
    cpu_before = psutil.cpu_percent(interval=None)

    t_inf0 = time.time()
    outputs = session.run(None, {input_name: x})
    t_inf1 = time.time()
    tempo = t_inf1 - t_inf0

    cpu_after = psutil.cpu_percent(interval=None)
    ram_after = get_ram_mb()

    print("=== RESULTADOS ===")
    print(f"Tempo inferência: {tempo:.3f} s")
    print(f"CPU uso (snapshot): before {cpu_before}%, after {cpu_after}%")
    print(f"RAM before: {ram_before:.2f} MB, after: {ram_after:.2f} MB")
    print("Outputs shapes:", [o.shape for o in outputs])

    # cleanup
    del session, outputs, x
    gc.collect()

if __name__ == '__main__':
    run_inference()       

