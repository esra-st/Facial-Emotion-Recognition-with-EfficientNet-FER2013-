# Facial Emotion Recognition with EfficientNet (FER2013) — Training + Real‑Time Webcam Demo

This project trains an **EfficientNetB0**-based classifier on the **FER2013** dataset (7 emotions) and runs a **real‑time webcam demo** using OpenCV face detection and the trained Keras model. citeturn0search2

## 1) Classes (7)
The class order must match training and inference:
`angry, disgust, fear, happy, neutral, sad, surprise` fileciteturn2file0L1-L20

---

## 2) Repository layout (recommended)

```
.
├─ app.py
├─ emotion_model.keras           # your exported trained model (see section 5)
├─ requirements.txt              # inference + basic utilities
├─ requirements-train.txt        # (optional) for training/evaluation notebook
└─ README.md
```

> If your model file has a different name, update `MODEL_PATH` in `app.py`. fileciteturn2file0L1-L20

---

## 3) Setup (local machine)

### 3.1 Create a virtual environment (recommended)

**Windows (PowerShell)**
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3.2 Install dependencies

**Inference (webcam demo):**
```bash
pip install -r requirements.txt
```

**Optional training/evaluation extras:**
```bash
pip install -r requirements-train.txt
```

---

## 4) Run the webcam demo

Place your trained model file next to `app.py`:
- `emotion_model.keras`

Then run:
```bash
python app.py
```

### Controls
- Press **Q** to quit. fileciteturn2file0L82-L131

### What the script does (pipeline)
1. Loads the Keras model and prints `model.input_shape` for verification. fileciteturn2file0L34-L52  
2. Opens the webcam (`cv2.VideoCapture(0)`) and reads frames. fileciteturn2file0L43-L63  
3. Detects faces using Haar Cascade (`haarcascade_frontalface_default.xml`). fileciteturn2file0L12-L18  
4. Chooses the **largest** detected face, adds padding, crops it. fileciteturn2file0L70-L94  
5. Preprocesses the crop:
   - BGR → RGB
   - resize to **224×224**
   - EfficientNet `preprocess_input` fileciteturn2file0L23-L33  
6. Runs prediction and overlays the label + confidence on the frame. fileciteturn2file0L95-L124  
7. Optional probability **smoothing** (EMA) to reduce label flicker (`USE_SMOOTHING=True`, `ALPHA=0.7`). fileciteturn2file0L19-L21  

### Camera troubleshooting
- If the camera does not open, try a different index:
  ```python
  cap = cv2.VideoCapture(1)  # or 2
  ```
- The script sets webcam resolution to 1280×720; you can edit it in `app.py`. fileciteturn2file0L57-L60

### Input-size mismatch troubleshooting
If you get an error like “expected shape (None, 224, 224, 3) …”:
- Your model expects a different size than the script uses.
- Fix by making them match:
  - either export the correct model, or
  - change the resize in `preprocess_face()` to the model’s expected size. fileciteturn2file0L23-L33

---

## 5) Exporting the model from the Kaggle notebook

In your Kaggle notebook, after training:
```python
model.save("/kaggle/working/emotion_model.keras")
```

Then from Kaggle:
1. Open the **Output** tab
2. Download `emotion_model.keras`
3. Put it next to `app.py` (or into your repo root)

> The linked notebook is hosted on Kaggle and uses the FER-2013 dataset. citeturn0search2

---

## 6) Training overview (Kaggle notebook)

Typical pipeline used:
- Dataset: FER2013 (7 emotions) citeturn0search2  
- Transfer learning with **EfficientNetB0 (ImageNet weights)**
- Data augmentation (flip/rotation/zoom)
- Class imbalance handling (class weights / boosted weights)
- Callbacks: EarlyStopping + ReduceLROnPlateau
- Two-stage training: head training (frozen backbone) → fine-tuning (partial unfreeze)

---

## 7) Known limitations
- FER2013 is challenging (low-res, noise), so some class confusions are expected.
- Real-time performance depends on CPU/GPU; smoothing can add a tiny latency.

---

## 8) License
Educational / academic use.

## 📘 Training Notebook

The model was trained on the FER-2013 dataset using EfficientNetB0.  
Full training pipeline is available here:

👉 **Kaggle Notebook:** https://www.kaggle.com/code/esra0706/facial-emotion-recognition-with-efficientnet

