from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import dlib
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# --- Configuración de Dlib ---
# Asegúrate de tener el archivo 'shape_predictor_68_face_landmarks.dat'
# en la misma carpeta donde está este app.py
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(predictor_path):
    raise FileNotFoundError("Falta el archivo shape_predictor_68_face_landmarks.dat")

predictor = dlib.shape_predictor(predictor_path)


@app.route('/')
def home():
    return "Servidor IA facial TEPF activo ✅ (Dlib edition)"


# --- Endpoint para procesar imágenes faciales ---
@app.route('/analizar', methods=['POST'])
def analizar():
    try:
        if 'imagen' not in request.files:
            return jsonify({"error": "No se recibió ninguna imagen"}), 400

        file = request.files['imagen']
        img = Image.open(file.stream).convert("RGB")
        frame = np.array(img)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        rostros = detector(gray)

        if len(rostros) == 0:
            return jsonify({"resultado": "No se detectó ningún rostro"}), 200

        # Dibujar landmarks
        for rostro in rostros:
            landmarks = predictor(gray, rostro)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Codificar la imagen procesada a base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        print(f"✅ Rostro detectado con 68 puntos faciales")

        return jsonify({
            "resultado": "Rostro detectado con 68 puntos faciales",
            "imagenProcesada": img_base64
        }), 200

    except Exception as e:
        print(f"🔥 Error interno en el servidor Flask: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("💡 Servidor IA facial TEPF iniciando en http://0.0.0.0:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=True)
