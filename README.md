
🎭 Real-Time Facial Emotion Recognition with EfficientNet

This project is a real-time facial emotion recognition system that uses a deep learning model based on EfficientNet to classify human facial emotions from a webcam feed.

The system detects faces using OpenCV Haar Cascades, preprocesses the detected face images, and predicts emotions using a trained TensorFlow/Keras model.

The training notebook used for model development is available here:
👉 Kaggle Notebook: https://www.kaggle.com/code/esra0706/facial-emotion-recognition-with-efficientnet


📌 Features

🎥 Real-time emotion recognition from webcam

🧠 EfficientNet-based deep learning model

🙂 Supports 7 emotion classes

🔍 Face detection using OpenCV Haar Cascade

📉 Prediction smoothing to reduce flickering

⚡ Runs locally with no internet requirement

😃 Supported Emotion Classes

The model predicts the following 7 facial emotions (order is critical and matches the training notebook):

angry
disgust
fear
happy
neutral
sad
surprise


These labels are defined in the application code and must match the model output exactly 

app

.

📂 Project Structure
├── app.py                         # Main application (real-time inference)
├── emotion_model.keras            # Trained EfficientNet model
├── facial-emotion-recognition-with-efficientnet.ipynb
│                                  # Model training & experimentation notebook
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation

🧠 How It Works
1️⃣ Face Detection

Uses OpenCV's haarcascade_frontalface_default.xml

Detects all faces in a frame

Selects the largest detected face for prediction

2️⃣ Preprocessing

Each detected face undergoes the same preprocessing steps used during training:

Convert BGR → RGB

Resize to 224 × 224

Normalize using EfficientNet preprocess_input

This ensures full consistency with the training pipeline 

app

.

3️⃣ Emotion Prediction

The processed face is passed to the trained EfficientNet model

The model outputs probabilities for each emotion class

The emotion with the highest probability is selected

4️⃣ Prediction Smoothing (Optional)

To reduce rapid emotion changes between frames, exponential smoothing is applied:

smooth_probs = ALPHA * previous + (1 - ALPHA) * current


ALPHA = 0.7

Can be disabled by setting USE_SMOOTHING = False

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/emotion-recognition-efficientnet.git
cd emotion-recognition-efficientnet

2️⃣ Install Dependencies
pip install -r requirements.txt


Dependencies include TensorFlow, OpenCV, and NumPy 

requirements

.

▶️ Running the Application

Make sure emotion_model.keras is in the project root directory.

python app.py

Controls

Q → Quit the application

Once started, the webcam feed will appear with:

A bounding box around the detected face

Predicted emotion label

Confidence score (probability)

🧪 Model Training

The model was trained using the notebook:

facial-emotion-recognition-with-efficientnet.ipynb


This notebook includes:

Dataset loading and preprocessing

EfficientNet model configuration

Training & evaluation

Model export to .keras format

⚠️ Important:
The preprocessing steps and class order in app.py must exactly match the notebook configuration, otherwise predictions will be incorrect.

🚀 Performance Tips

Ensure good lighting for better face detection

Use a webcam with at least 720p resolution

Avoid extreme face angles

One face at a time yields best results

📌 Notes & Limitations

Works best on frontal faces

Haar Cascade may miss faces at extreme angles

Model accuracy depends on training dataset diversity

Designed for demo & research purposes

📜 License

This project is intended for educational and research use.
You are free to modify and extend it.
