import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ✅ Load model
model = load_model("emergency_model.h5", compile=False)

# ✅ Classes (same order as training)
classes = ['accident', 'call', 'doctor', 'help', 'hot', 'lose', 'pain', 'thief']

# 🎥 Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 🔄 Preprocess
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.reshape(img, (1, 64, 64, 3))

    # 🤖 Prediction
    prediction = model.predict(img)[0]

    # 🔥 Get top 2 predictions
    top2 = np.argsort(prediction)[-2:]
    best = top2[-1]
    second = top2[-2]

    confidence = prediction[best]
    difference = prediction[best] - prediction[second]

    # 🧠 Smart decision
    if confidence < 0.7 or difference < 0.2:
        label = "Confused"
    else:
        label = classes[best]

    # 🖥️ Display
    text = f"{label} ({confidence:.2f})"

    cv2.putText(frame, text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Emergency Sign Detection", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()