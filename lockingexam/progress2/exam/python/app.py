from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from simple_facerec_with_similarity import SimpleFacerecWithSimilarity

app = Flask(__name__)

# Inisialisasi SimpleFacerecWithSimilarity
sfr = SimpleFacerecWithSimilarity()
sfr.load_encoding_images("C:\\Users\\User\\Desktop\\lockingexam\\progress2\\exam\\images\\")

# Endpoint untuk halaman utama
@app.route('/')
def home():
    return render_template('home.html')

# Endpoint untuk memulai deteksi wajah
@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    # Menerima gambar dari permintaan POST
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Melakukan deteksi wajah dan mendapatkan hasilnya
    face_locations, face_names, face_similarities = sfr.detect_known_faces(image)

    # Menggabungkan hasil menjadi format JSON
    results = []
    for face_loc, name, similarity in zip(face_locations, face_names, face_similarities):
        result = {
            'name': name,
            'similarity': similarity,
            'location': face_loc.tolist()
        }
        results.append(result)

    return jsonify(results)

if __name__ == '__main__':
    app.run()