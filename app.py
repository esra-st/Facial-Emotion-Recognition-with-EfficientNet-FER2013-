import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = "emotion_model.keras"

# NOTEBOOK'TAKİ SIRA (çok önemli!)
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# titreşim azaltma
USE_SMOOTHING = True
ALPHA = 0.7

def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """
    Notebook ile aynı:
    - RGB
    - 224x224
    - preprocess_input
    """
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32)
    x = np.expand_dims(x, axis=0)         # (1,224,224,3)
    x = preprocess_input(x)               # NOTEBOOK'TAKİ İŞLEM
    return x

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")

    print("Model yükleniyor...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model yüklendi.")
    print("Model input shape:", model.input_shape)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Kamera açılamadı.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smooth_probs = None

    print("🎥 Kamera başladı | Çıkış: Q")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) > 0:
            # en büyük yüz
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

            # biraz büyüt
            pad = int(0.15 * w)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)

            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                x_in = preprocess_face(face)
                probs = model.predict(x_in, verbose=0)[0]  # (7,)

                if USE_SMOOTHING:
                    if smooth_probs is None:
                        smooth_probs = probs
                    else:
                        smooth_probs = ALPHA * smooth_probs + (1 - ALPHA) * probs
                    p = smooth_probs
                else:
                    p = probs

                idx = int(np.argmax(p))
                conf = float(p[idx])
                label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} ({conf:.2f})",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                # İstersen top-3 debug (angry takılması var mı görürüz)
                top3 = np.argsort(p)[-3:][::-1]
                # print("TOP3:", [(CLASS_NAMES[i], float(p[i])) for i in top3])

        cv2.imshow("Emotion Recognition (EfficientNet)", frame)
        if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
