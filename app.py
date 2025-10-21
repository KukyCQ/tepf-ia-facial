from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

# --- Configuración básica de Mediapipe ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

@app.route('/')
def home():
    return "Servidor IA facial TEPF activo ✅"

# --- Endpoint para procesar imágenes faciales ---
@app.route('/analizar', methods=['POST'])
def analizar():
    try:
        if 'imagen' not in request.files:
            return jsonify({"error": "No se recibió ninguna imagen"}), 400

        file = request.files['imagen']
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "La imagen no es válida o está corrupta"}), 400

        # Procesamiento con Mediapipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(img_rgb)

        if not result.multi_face_landmarks:
            return jsonify({"resultado": "No se detectó ningún rostro"}), 200

        # Dibujar los puntos y malla facial
        annotated = img.copy()
        for face_landmarks in result.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

        # Convertir a JPG y luego a base64
        _, buffer = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        cantidad = len(result.multi_face_landmarks[0].landmark)
        print(f"✅ Rostro detectado con {cantidad} puntos faciales.")

        return jsonify({
            "resultado": f"Rostro detectado con {cantidad} puntos faciales",
            "imagenProcesada": img_base64
        }), 200

    except Exception as e:
        print(f"🔥 Error interno en el servidor Flask: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("💡 Servidor IA facial TEPF iniciando en http://0.0.0.0:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=True)
