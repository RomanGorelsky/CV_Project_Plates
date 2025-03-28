import cv2
import sqlite3
import time

import numpy as np
from inference import load_keras_model, load_config_from_yaml, postprocess_model_output
import keras

# Path to CNN model
model_path = r"cnn_ocr-epoch_728-acc_0.922.keras"
# Path to config file
config_file = r"config.yaml"

# Load config file
config = load_config_from_yaml(config_file)

# Load CNN model
model = load_keras_model(
    model_path, vocab_size=config.vocabulary_size, max_plate_slots=config.max_plate_slots
)

# Conenction to SQLite server
conn = sqlite3.connect('car_db.sqlite')
cursor = conn.cursor()

# Parameters for VideoCapture
cap = cv2.VideoCapture(0)
last_check = time.time()
check_interval = 5  # seconds
label = ""
label_display_time = 3  # show label for 3 seconds
label_shown_at = None

# Create empty table
cursor.execute("""
CREATE TABLE IF NOT EXISTS database (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT UNIQUE
)
""")
conn.commit()

# Fill table with info from database.txt
with open('database.txt', 'r') as f:
    for line in f:
        plate = line.strip()
        if plate:
            try:
                cursor.execute("INSERT INTO database (plate) VALUES (?)", (plate,))
            except sqlite3.IntegrityError:
                pass 
conn.commit()


while True:
    # Read from web_camera
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Perform OCR check every 5 seconds
    if current_time - last_check >= check_interval:
        last_check = current_time
        label_shown_at = current_time
        label = "No plate detected"  # default label

        # OCR text detection
        frame_2 = cv2.resize(frame, (140,70))
        frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
        frame_2 = np.expand_dims(frame_2, 0)
        prediction = model(frame_2, training=False)
        prediction = keras.ops.stop_gradient(prediction).numpy()
        plate, probs = postprocess_model_output(
            prediction=prediction,
            alphabet=config.alphabet,
            max_plate_slots=config.max_plate_slots,
            vocab_size=config.vocabulary_size,
        )
        print(plate)
        text = plate
        text = text.replace('_', "")

        # Check that correct text was found
        if len(text) >= 5:
            # Check in DB
            cursor.execute("SELECT EXISTS(SELECT 1 FROM database WHERE plate=?)", (text,))
            found = cursor.fetchone()[0]
            label = f"Plate: {text} | {'Found' if found else 'Not found'}"

    if label and label_shown_at and (current_time - label_shown_at > label_display_time):
        label = ""
        label_shown_at = None

    if label:
        cv2.putText(frame, label, (30, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if "Found" in label else (0, 0, 255), 2)

    cv2.imshow("Live OCR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()