import cv2
from simple_facerec_with_similarity import SimpleFacerecWithSimilarity
#import ujian

# Encode wajah dari folder
sfr = SimpleFacerecWithSimilarity()
sfr.load_encoding_images(r"C:\\Users\\User\\Desktop\\lockingexam\\progress2\\exam\\images\\")

# Muat Kamera
cap = cv2.VideoCapture(0)  # Ubah indeks kamera menjadi 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Deteksi Wajah
    face_locations, face_names, face_similarities = sfr.detect_known_faces(frame)

    for face_loc, name, similarity in zip(face_locations, face_names, face_similarities):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, f"{name} ({similarity:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()