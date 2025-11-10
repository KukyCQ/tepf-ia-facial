from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import urllib.request
import threading
import time
import sys
import types
import traceback

# === Parche para evitar la carga de OpenCV (cv2) en entornos sin GUI como Render ===
sys.modules['cv2'] = types.ModuleType('cv2')
setattr(sys.modules['cv2'], '__version__', 'stub')

import mediapipe as mp

# ==== Configuración base ====
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "https://soporte-tepf-cf23c.web.app",
    "http://localhost:5000",
    "http://127.0.0.1:5000"
]}})

# ==== Inicialización global de MediaPipe ====
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

@app.route('/')
def home():
    return "Servidor IA facial TEPF activo ✅ (MediaPipe FaceMesh + análisis regional)"

# ==== Funciones auxiliares ====
def calcular_simetria_region(puntos, w, h):
    puntos = np.array([(lm.x * w, lm.y * h) for lm in puntos], dtype=np.float32)
    eje_central = np.mean(puntos[:, 0])
    izquierda = puntos[puntos[:, 0] < eje_central]
    derecha = puntos[puntos[:, 0] > eje_central]
    if len(izquierda) == 0 or len(derecha) == 0:
        return 0.0
    derecha_ref = derecha.copy()
    derecha_ref[:, 0] = 2 * eje_central - derecha[:, 0]
    n = min(len(izquierda), len(derecha_ref))
    if n == 0:
        return 0.0
    diff = np.linalg.norm(izquierda[:n] - derecha_ref[:n], axis=1).mean()
    return max(0.0, min(100.0, 100.0 - diff / 2.0))

def calcular_simetria_total(landmarks, w, h):
    regiones = {
        "ojos": list(range(33, 133)),
        "nariz": list(range(1, 10)) + list(range(168, 197)),
        "boca": list(range(78, 308)),
        "rostro": list(range(0, 468))
    }

    resultados = {}
    for nombre, idxs in regiones.items():
        puntos = [landmarks[i] for i in idxs if i < len(landmarks)]
        if puntos:
            resultados[nombre] = calcular_simetria_region(puntos, w, h)
        else:
            resultados[nombre] = 0

    sim_total = (
        resultados["ojos"] * 0.25 +
        resultados["nariz"] * 0.20 +
        resultados["boca"] * 0.35 +
        resultados["rostro"] * 0.20
    )
    return round(sim_total, 2), resultados

# ==== Endpoint principal ====
@app.route('/analizar', methods=['POST'])
def analizar():
    try:
        if 'imagen' not in request.files:
            return jsonify({"error": "No se recibió ninguna imagen"}), 400

        file = request.files['imagen']
        pil = Image.open(file.stream).convert("RGB")
        frame = np.array(pil)

        # Validar formato y tamaño del frame
        if frame is None or frame.size == 0:
            raise ValueError("La imagen recibida está vacía o no es válida.")

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("Dimensiones de imagen inválidas.")

        # Procesar con MediaPipe
        results = face_mesh.process(frame)

        if not results.multi_face_landmarks:
            print("⚠️ No se detectó ningún rostro.")
            return jsonify({"resultado": "No se detectó ningún rostro", "simetria": 0}), 200

        fl = results.multi_face_landmarks[0]
        sim_total, regiones = calcular_simetria_total(fl.landmark, w, h)

        # ✅ Convertir todo a tipos nativos (para evitar el error JSON)
        regiones_py = {k: float(v) for k, v in regiones.items()}
        sim_total_py = float(sim_total)

        print(f"✅ Simetría total: {sim_total_py}% | Detalle: {regiones_py}")
        return jsonify({
            "resultado": f"Simetría facial estimada: {sim_total_py}%",
            "simetria": sim_total_py,
            "detalles": regiones_py
        }), 200

    except Exception as e:
        print("🔥 Error interno en /analizar:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ==== Keep-alive para Render ====
def keep_alive():
    while True:
        try:
            urllib.request.urlopen("https://tepf-ia-facial.onrender.com/")
            print("🔄 Keep-alive ping exitoso.")
        except Exception as e:
            print(f"⚠️ Keep-alive falló: {e}")
        time.sleep(600)

threading.Thread(target=keep_alive, daemon=True).start()
print("🌀 Keep-alive activado para Render cada 10 minutos.")

if __name__ == '__main__':
    print("💡 Servidor IA facial (MediaPipe + análisis facial) iniciando en http://0.0.0.0:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False)
