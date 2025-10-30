from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import dlib
import numpy as np
import base64
from PIL import Image
import os
import urllib.request
import threading
import time

app = Flask(__name__)

# ✅ Permite llamadas solo desde tus dominios Firebase y localhost
CORS(app, resources={r"/*": {"origins": [
    "https://soporte-tepf-cf23c.web.app",
    "http://localhost:5000",
    "http://127.0.0.1:5000"
]}})

# ==== CONFIGURACIÓN DEL MODELO DLIB ====
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(MODEL_PATH):
    print("🔽 Descargando modelo desde Firebase Storage...")
    url = "https://firebasestorage.googleapis.com/v0/b/soporte-tepf-cf23c.firebasestorage.app/o/shape_predictor_68_face_landmarks.dat?alt=media&token=de68a62f-b70d-4feb-8c34-3aa4d05e8da2"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("✅ Modelo descargado correctamente")

# Inicializar detector y predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)


@app.route('/')
def home():
    return "Servidor IA facial TEPF activo ✅ (Simetría facial avanzada con CORS habilitado)"


# ==== FUNCIÓN DE CÁLCULO DE SIMETRÍA ====
def calcular_simetria(landmarks):
    puntos = np.array([[p.x, p.y] for p in landmarks.parts()])
    eje_central = np.mean(puntos[:, 0])

    izquierda = puntos[puntos[:, 0] < eje_central]
    derecha = puntos[puntos[:, 0] > eje_central]

    if len(izquierda) == 0 or len(derecha) == 0:
        return 0

    derecha_reflejada = derecha.copy()
    derecha_reflejada[:, 0] = 2 * eje_central - derecha[:, 0]

    n = min(len(izquierda), len(derecha_reflejada))
    diferencia = np.mean(np.linalg.norm(izquierda[:n] - derecha_reflejada[:n], axis=1))
    diferencia_norm = max(0, min(100, 100 - diferencia / 2))
    return round(diferencia_norm, 2)


# ==== ENDPOINT PRINCIPAL ====
@app.route('/analizar', methods=['POST'])
def analizar():
    try:
        if 'imagen' not in request.files:
            return jsonify({"error": "No se recibió ninguna imagen"}), 400

        # Leer la imagen y convertir a formato BGR (nativo de OpenCV)
        file = request.files['imagen']
        pil_img = Image.open(file.stream).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rostros = detector(gray)
        if len(rostros) == 0:
            print("⚠️ No se detectó ningún rostro.")
            return jsonify({"resultado": "No se detectó ningún rostro"}), 200

        simetria_promedio = 0
        for rostro in rostros:
            landmarks = predictor(gray, rostro)
            simetria_promedio = calcular_simetria(landmarks)

            # Dibujar contorno del rostro
            x, y, w, h = rostro.left(), rostro.top(), rostro.width(), rostro.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)

            # Dibujar puntos faciales (verde brillante)
            for n in range(0, 68):
                px, py = landmarks.part(n).x, landmarks.part(n).y
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

        # ✅ Codificar imagen procesada sin alterar canales (mantener BGR)
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            raise Exception("No se pudo codificar la imagen procesada")

        img_base64 = base64.b64encode(buffer).decode('utf-8')
        print(f"✅ Simetría facial detectada: {simetria_promedio}% (puntos visibles)")

        return jsonify({
            "resultado": f"Simetría facial estimada: {simetria_promedio}%",
            "simetria": simetria_promedio,
            "imagenProcesada": img_base64
        }), 200

    except Exception as e:
        print(f"🔥 Error interno: {e}")
        return jsonify({"error": str(e)}), 500


# ==== MANTENER VIVA LA INSTANCIA (Render auto-sleep fix) ====
def keep_alive():
    while True:
        try:
            urllib.request.urlopen("https://tepf-ia-facial.onrender.com/")
        except Exception as e:
            print(f"⚠️ Keep-alive falló: {e}")
        time.sleep(600)  # cada 10 min

threading.Thread(target=keep_alive, daemon=True).start()


if __name__ == '__main__':
    print("💡 Servidor IA facial TEPF iniciando en http://0.0.0.0:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False)
