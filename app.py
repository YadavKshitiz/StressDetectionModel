# import os
# import numpy as np
# import base64
# import joblib  # For loading the scaler
# import librosa  # For audio feature extraction
# import cv2  # For image decoding and resizing
# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from io import BytesIO
# import time  # For timing the prediction

# app = Flask(__name__)

# # --- CONFIGURATION (MUST MATCH YOUR TRAINING DATA) ---
# # Image settings
# IMG_HEIGHT, IMG_WIDTH = 48, 48  # ✅ Updated to match facial_model input shape
# # Audio settings
# SR = 22050  # Sample Rate for Librosa
# N_MFCC = 40  # Number of MFCCs to extract
# MAX_PAD_LENGTH = 174  # Must match your audio_model input shape

# # --- 1. MODEL & ARTIFACT LOADING ---
# try:
#     FACIAL_MODEL = load_model('facial_model.h5')
#     AUDIO_MODEL = load_model('audio_model.h5')
#     SURVEY_MODEL = load_model('survey_model.h5')
#     SURVEY_SCALER = joblib.load('survey_scaler.pkl')

#     C2_CATEGORIES = np.load('C2_values.npy', allow_pickle=True)
#     C3_CATEGORIES = np.load('C3_values.npy', allow_pickle=True)
#     C4_CATEGORIES = np.load('C4_values.npy', allow_pickle=True)
#     C6_CATEGORIES = np.load('C6_values.npy', allow_pickle=True)

#     print("--- Calmi Stress Detector API Ready ---")
#     print(f"Models and {SURVEY_SCALER.n_features_in_}-feature scaler loaded.")

# except Exception as e:
#     print(f"FATAL ERROR: Could not load required artifacts: {e}")
#     exit(1)


# # --- 2. PREPROCESSING FUNCTIONS ---

# def preprocess_image(base64_img_string):
#     """Decodes Base64 image, converts to grayscale, resizes, and normalizes for facial_model."""
#     try:
#         img_bytes = base64.b64decode(base64_img_string)
#         nparr = np.frombuffer(img_bytes, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         if img is None:
#             raise ValueError("Could not decode image. Image is None.")

#         # Convert to grayscale since model expects 1 channel
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Resize to model input size (48x48)
#         img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

#         # Normalize to [0,1]
#         img = img.astype('float32') / 255.0

#         # Add channel and batch dimensions -> (1, 48, 48, 1)
#         img = np.expand_dims(img, axis=-1)
#         img = np.expand_dims(img, axis=0)

#         return img

#     except Exception as e:
#         print(f"Image preprocessing failed: {e}")
#         return np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 1), dtype='float32')


# def preprocess_audio(base64_audio_string):
#     """Decodes Base64 audio, extracts MFCC features, and pads/truncates."""
#     try:
#         audio_bytes = base64.b64decode(base64_audio_string)
#         y, sr = librosa.load(BytesIO(audio_bytes), sr=SR)
#         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

#         if mfccs.shape[1] > MAX_PAD_LENGTH:
#             mfccs = mfccs[:, :MAX_PAD_LENGTH]
#         elif mfccs.shape[1] < MAX_PAD_LENGTH:
#             pad_width = MAX_PAD_LENGTH - mfccs.shape[1]
#             mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

#         mfccs_input = np.expand_dims(mfccs, axis=-1)
#         mfccs_input = np.expand_dims(mfccs_input, axis=0)
#         return mfccs_input.astype('float32')

#     except Exception as e:
#         print(f"Audio preprocessing failed: {e}")
#         return np.zeros((1, N_MFCC, MAX_PAD_LENGTH, 1), dtype='float32')


# def preprocess_survey(answers: list):
#     """Converts a list of 5 survey answers into a scaled numerical vector."""
#     if len(answers) != 5:
#         print(f"Error: Expected 5 survey answers, got {len(answers)}.")
#         return np.zeros((1, SURVEY_SCALER.n_features_in_), dtype='float32')

#     category_map = {
#         0: list(C2_CATEGORIES),
#         1: list(C3_CATEGORIES),
#         2: list(C4_CATEGORIES),
#         3: list(C6_CATEGORIES),
#         4: ["Very Energetic", "Somewhat Energetic", "Neutral", "Tired"]
#     }

#     ohe_vector = []
#     for i, answer in enumerate(answers):
#         cats = category_map.get(i)
#         try:
#             vector = [0] * len(cats)
#             index = cats.index(answer)
#             vector[index] = 1
#             ohe_vector.extend(vector)
#         except Exception:
#             print(f"Warning: Invalid answer '{answer}' for Q{i}. Using zeros.")
#             ohe_vector.extend([0] * len(cats))

#     final_features = np.array(ohe_vector).reshape(1, -1)
#     if final_features.shape[1] != SURVEY_SCALER.n_features_in_:
#         print(f"Scaler mismatch: expected {SURVEY_SCALER.n_features_in_}, got {final_features.shape[1]}")
#         return np.zeros((1, SURVEY_SCALER.n_features_in_), dtype='float32')

#     scaled_features = SURVEY_SCALER.transform(final_features)
#     return scaled_features.astype('float32')


# # --- 3. API ENDPOINT ---
# @app.route('/predict_stress', methods=['POST'])
# def predict_stress():
#     start_time = time.time()

#     if not request.json:
#         return jsonify({"error": "No data received. Expected JSON payload with image_b64, audio_b64, and survey_answers."}), 400

#     try:
#         data = request.json
#         processed_img = preprocess_image(data.get('image_b64', ''))
#         processed_aud = preprocess_audio(data.get('audio_b64', ''))
#         processed_survey = preprocess_survey(data.get('survey_answers', []))

#         image_features = FACIAL_MODEL.predict(processed_img, verbose=0)
#         audio_features = AUDIO_MODEL.predict(processed_aud, verbose=0)

#         fusion_input = np.concatenate([
#             image_features,
#             audio_features,
#             processed_survey
#         ], axis=1)

#         final_prediction = SURVEY_MODEL.predict(fusion_input, verbose=0)
#         stress_prob = float(final_prediction[0][0])
#         stress_result = "High Stress" if stress_prob >= 0.5 else "Low Stress"

#         latency = round((time.time() - start_time) * 1000, 2)

#         return jsonify({
#             "status": "success",
#             "stress_level": stress_result,
#             "probability": round(stress_prob, 4),
#             "latency_ms": latency
#         })

#     except Exception as e:
#         print(f"Prediction failed: {e}")
#         return jsonify({
#             "error": "Internal server error during prediction",
#             "details": str(e)
#         }), 500


# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 8080))
#     app.run(host='0.0.0.0', port=port)
























import os
import numpy as np
import base64
import joblib  # For loading the scaler
import librosa  # For audio feature extraction
import cv2  # For image decoding and resizing
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from io import BytesIO
import time  # For timing the prediction

app = Flask(__name__)

# --- CONFIGURATION (MUST MATCH YOUR TRAINING DATA) ---
# Image settings
IMG_HEIGHT, IMG_WIDTH = 48, 48  # ✅ Facial model input shape
# Audio settings
SR = 22050  # Sample Rate for Librosa
N_MFCC = 40  # Number of MFCCs to extract
MAX_PAD_LENGTH = None  # ✅ Will be auto-detected from AUDIO_MODEL

# --- 1. MODEL & ARTIFACT LOADING ---
try:
    FACIAL_MODEL = load_model('facial_model.h5')
    AUDIO_MODEL = load_model('audio_model.h5')
    SURVEY_MODEL = load_model('survey_model.h5')
    SURVEY_SCALER = joblib.load('survey_scaler.pkl')

    # ✅ Auto-detect MFCC shape from AUDIO_MODEL
    _, n_mfcc, max_pad_len, _ = AUDIO_MODEL.input_shape
    MAX_PAD_LENGTH = int(max_pad_len)
    print(f"✅ Auto-detected MAX_PAD_LENGTH = {MAX_PAD_LENGTH} from audio_model.h5")

    # Load category mappings
    C2_CATEGORIES = np.load('C2_values.npy', allow_pickle=True)
    C3_CATEGORIES = np.load('C3_values.npy', allow_pickle=True)
    C4_CATEGORIES = np.load('C4_values.npy', allow_pickle=True)
    C6_CATEGORIES = np.load('C6_values.npy', allow_pickle=True)

    print("--- Calmi Stress Detector API Ready ---")
    print(f"Models and {SURVEY_SCALER.n_features_in_}-feature scaler loaded.")

except Exception as e:
    print(f"FATAL ERROR: Could not load required artifacts: {e}")
    exit(1)


# --- 2. PREPROCESSING FUNCTIONS ---

def preprocess_image(base64_img_string):
    """Decodes Base64 image, converts to grayscale, resizes, and normalizes for facial_model."""
    try:
        img_bytes = base64.b64decode(base64_img_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image. Image is None.")

        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype('float32') / 255.0

        # Add channel and batch dims -> (1, 48, 48, 1)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        return img

    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        return np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 1), dtype='float32')


def preprocess_audio(base64_audio_string):
    """Decodes Base64 audio, extracts MFCC features, and pads/truncates."""
    try:
        audio_bytes = base64.b64decode(base64_audio_string)
        y, sr = librosa.load(BytesIO(audio_bytes), sr=SR)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

        # Adjust to match model’s input size
        if mfccs.shape[1] > MAX_PAD_LENGTH:
            mfccs = mfccs[:, :MAX_PAD_LENGTH]
        elif mfccs.shape[1] < MAX_PAD_LENGTH:
            pad_width = MAX_PAD_LENGTH - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

        mfccs_input = np.expand_dims(mfccs, axis=-1)
        mfccs_input = np.expand_dims(mfccs_input, axis=0)
        return mfccs_input.astype('float32')

    except Exception as e:
        print(f"Audio preprocessing failed: {e}")
        return np.zeros((1, N_MFCC, MAX_PAD_LENGTH, 1), dtype='float32')


def preprocess_survey(answers: list):
    """Converts a list of 5 survey answers into a scaled numerical vector."""
    if len(answers) != 5:
        print(f"Error: Expected 5 survey answers, got {len(answers)}.")
        return np.zeros((1, SURVEY_SCALER.n_features_in_), dtype='float32')

    category_map = {
        0: list(C2_CATEGORIES),
        1: list(C3_CATEGORIES),
        2: list(C4_CATEGORIES),
        3: list(C6_CATEGORIES),
        4: ["Very Energetic", "Somewhat Energetic", "Neutral", "Tired"]
    }

    ohe_vector = []
    for i, answer in enumerate(answers):
        cats = category_map.get(i)
        try:
            vector = [0] * len(cats)
            index = cats.index(answer)
            vector[index] = 1
            ohe_vector.extend(vector)
        except Exception:
            print(f"Warning: Invalid answer '{answer}' for Q{i}. Using zeros.")
            ohe_vector.extend([0] * len(cats))

    final_features = np.array(ohe_vector).reshape(1, -1)
    if final_features.shape[1] != SURVEY_SCALER.n_features_in_:
        print(f"Scaler mismatch: expected {SURVEY_SCALER.n_features_in_}, got {final_features.shape[1]}")
        return np.zeros((1, SURVEY_SCALER.n_features_in_), dtype='float32')

    scaled_features = SURVEY_SCALER.transform(final_features)
    return scaled_features.astype('float32')


# --- 3. API ENDPOINT ---
@app.route('/predict_stress', methods=['POST'])
def predict_stress():
    start_time = time.time()

    if not request.json:
        return jsonify({"error": "No data received. Expected JSON payload with image_b64, audio_b64, and survey_answers."}), 400

    try:
        data = request.json
        processed_img = preprocess_image(data.get('image_b64', ''))
        processed_aud = preprocess_audio(data.get('audio_b64', ''))
        processed_survey = preprocess_survey(data.get('survey_answers', []))

        image_features = FACIAL_MODEL.predict(processed_img, verbose=0)
        audio_features = AUDIO_MODEL.predict(processed_aud, verbose=0)

        fusion_input = np.concatenate([
            image_features,
            audio_features,
            processed_survey
        ], axis=1)

        final_prediction = SURVEY_MODEL.predict(fusion_input, verbose=0)
        stress_prob = float(final_prediction[0][0])
        stress_result = "High Stress" if stress_prob >= 0.5 else "Low Stress"

        latency = round((time.time() - start_time) * 1000, 2)

        return jsonify({
            "status": "success",
            "stress_level": stress_result,
            "probability": round(stress_prob, 4),
            "latency_ms": latency
        })

    except Exception as e:
        print(f"Prediction failed: {e}")
        return jsonify({
            "error": "Internal server error during prediction",
            "details": str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
