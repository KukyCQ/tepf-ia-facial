from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from PIL import Image
import urllib.request
import threading
import time
import mediapipe as mp

app = Flask(__name__)
# CORS: tus dominios
CORS(app, resources={r"/*": {"origins": [
    "https://soporte-tepf-cf23c.web.app",
    "http://localhost:5000",
    "http://127.0.0.1:5000"
]}})

# ==== MediaPipe FaceMesh (global, para reusar y que sea más rápido) ====
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# static_image_mode=True porque procesamos imágenes sueltas (no video)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,        # +precisión (iris y contornos finos)
    min_detection_confidence=0.5
)

@app.route('/')
def home():
    return "Servidor IA facial TEPF activo ✅ (MediaPipe FaceMesh + CORS)"

def calcular_simetria_facemesh(landmarks, w, h):
    """
    landmarks: lista de 468 puntos normalizados (x,y,z) en [0,1]
    Convertimos a píxeles y comparamos izquierda vs derecha reflejada respecto
    al eje X medio del rostro.
    """
    puntos = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)
    eje_central = np.mean(puntos[:, 0])  # promedio de X

    # separa por eje
    izquierda = puntos[puntos[:, 0] < eje_central]
    derecha = puntos[puntos[:, 0] > eje_central]
    if len(izquierda) == 0 or len(derecha) == 0:
        return 0.0

    # refleja derecha sobre el eje para comparar
    derecha_ref = derecha.copy()
    derecha_ref[:, 0] = 2 * eje_central - derecha[:, 0]

    n = min(len(izquierda), len(derecha_ref))
    if n == 0:
        return 0.0

    # distancia media punto a punto (no es pareo 1-1 anatómico, pero robusto estadísticamente)
    diff = np.linalg.norm(izquierda[:n] - derecha_ref[:n], axis=1).mean()

    # escala a 0–100 (ajuste suave empírico)
    sim = max(0.0, min(100.0, 100.0 - diff / 2.0))
    return round(float(sim), 2)

@app.route('/analizar', methods=['POST'])
def analizar():
    try:
        if 'imagen' not in request.files:
            return jsonify({"error": "No se recibió ninguna imagen"}), 400

        # Leer imagen -> BGR
        file = request.files['imagen']
        pil = Image.open(file.stream).convert("RGB")
        frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        h, w = frame.shape[:2]

        # MediaPipe procesa en RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            print("⚠️ No se detectó ningún rostro.")
            return jsonify({"resultado": "No se detectó ningún rostro"}), 200

        # Solo 1 rostro (max_num_faces=1)
        fl = results.multi_face_landmarks[0]
        simetria = calcular_simetria_facemesh(fl.landmark, w, h)

        # Dibujo: caja aproximada + malla
        xs = [lm.x * w for lm in fl.landmark]
        ys = [lm.y * h for lm in fl.landmark]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)

        mp_draw.draw_landmarks(
            image=frame,
            landmark_list=fl,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=1)
        )

        # Codificar para enviar al frontend
        _, buf = cv2.imencode('.jpg', frame)
        img_b64 = base64.b64encode(buf).decode('utf-8')

        print(f"✅ Simetría facial (FaceMesh): {simetria}%")
        return jsonify({
            "resultado": f"Simetría facial estimada: {simetria}%",
            "simetria": simetria,
            "imagenProcesada": img_b64
        }), 200

    except Exception as e:
        print(f"🔥 Error interno: {e}")
        return jsonify({"error": str(e)}), 500

# ==== Keep-alive (Render) ====
def keep_alive():
    while True:
        try:
            urllib.request.urlopen("https://tepf-ia-facial.onrender.com/")
        except Exception as e:
            print(f"⚠️ Keep-alive falló: {e}")
        time.sleep(600)

threading.Thread(target=keep_alive, daemon=True).start()

if __name__ == '__main__':
    print("💡 Servidor IA facial (MediaPipe) iniciando en http://0.0.0.0:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False)
