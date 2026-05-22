# Sign Language Interpreter

A real-time hand sign and gesture interpreter built using Streamlit, MediaPipe, and PyTorch. 

Instead of passing heavy raw camera frames through a resource-heavy CNN, this app uses Google MediaPipe to locate and extract 3D coordinates of 21 key hand landmarks, normalizes them relative to the wrist, and runs them through a lightweight 3-layer PyTorch MLP.

This approach makes inference extremely fast, light enough to run in a web browser, and completely invariant to lighting or background noise.

Live App Link: **[sign-language-detector-x5lfxjvkukvuryju4s46np.streamlit.app](https://sign-language-detector-x5lfxjvkukvuryju4s46np.streamlit.app/)**

---

## Features
- **Real-time Detection**: Capture camera input right inside the Streamlit web dashboard.
- **Skeletal Overlay**: Draws the tracked 21 landmarks and hand connections.
- **Robust Fallback Engine**: If no custom model weights are found, the app uses geometric finger-extension rules (checking finger tips vs joint heights) so it works perfectly out-of-the-box.
- **Probability Breakdown**: Shows real-time classification confidences using interactive Plotly gauge and bar charts.

---

## Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/viveksharma151/sign-language-detector.git
   cd sign-language-detector
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## Model & Custom Training

The PyTorch neural network is defined in `train.py` as a feedforward MLP:
- **Input**: 63 features (21 landmarks x 3 coordinates: X, Y, Z)
- **Hidden Layers**: 128 nodes (ReLU, Dropout 0.3) -> 64 nodes (ReLU, Dropout 0.3)
- **Output**: 8 classes (A, B, C, L, V, Y, Thumbs Up, Open Hand)

If you want to record custom gestures and retrain the model:
1. Run a script to gather coordinate arrays using MediaPipe.
2. Save them under `recorded_data/features.npy` and `recorded_data/labels.npy`.
3. Train the model:
   ```bash
   python train.py
   ```
4. The script will save `gesture_model.pth`. When you run `app.py`, it detects this file and switches from the fallback engine to the trained PyTorch model.
