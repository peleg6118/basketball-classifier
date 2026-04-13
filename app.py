"""
app.py  –  Flask backend for Basketball Player Classifier
-----------------------------------------------------------
Run:
    python app.py

Endpoints:
    GET  /           → serve the frontend (index.html)
    POST /predict    → accept JSON player stats, return classification
    GET  /health     → health check
"""

import pickle
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template

# Try to import TensorFlow (optional at startup; required for prediction)
try:
    from tensorflow.keras.models import load_model as tf_load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR / 'model'
MODEL_PATH = MODEL_DIR / 'nn_model.keras'
SCALER_PATH= MODEL_DIR / 'scaler.pkl'
MEANS_PATH = MODEL_DIR / 'means.pkl'

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder='templates')

# Lazy-load model & scaler once
_model  = None
_scaler = None
_means  = None

def _load_artifacts():
    global _model, _scaler, _means
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run `python train_model.py --data <your_csv>` first."
            )
        _model  = tf_load_model(str(MODEL_PATH))
        with open(SCALER_PATH, 'rb') as f:
            _scaler = pickle.load(f)
        with open(MEANS_PATH, 'rb') as f:
            _means  = pickle.load(f)


# Feature order must match training exactly
FEATURES = [
    'log_MIN', '3PM', '3PA',
    'log_FTM', 'FTA', 'log_TOV', 'PF',
    'log_ORB', 'log_REB', 'AST',
    'Stocks', 'height_cm', 'weight_kg'
]

LABELS = {
    0: {'he': 'שחקן טוב',      'en': 'Good Player',    'emoji': '🌟', 'color': '#F5A623'},
    1: {'he': 'שחקן ממוצע',    'en': 'Average Player', 'emoji': '⚡', 'color': '#4A90D9'},
    2: {'he': 'שחקן לא טוב',   'en': 'Below Average',  'emoji': '📉', 'color': '#E74C3C'},
}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_ready': MODEL_PATH.exists()})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        _load_artifacts()
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 503

    data = request.get_json(force=True)

    # ── Required raw inputs from the form ──────────────────────────────────
    try:
        MIN_val = float(data['MIN'])         # minutes per game
        PM3     = float(data['3PM'])          # 3-pointers made
        PA3     = float(data['3PA'])          # 3-pointers attempted
        FTM     = float(data['FTM'])          # free throws made
        FTA     = float(data['FTA'])          # free throws attempted
        TOV     = float(data['TOV'])          # turnovers
        PF      = float(data['PF'])           # personal fouls
        ORB     = float(data['ORB'])          # offensive rebounds
        REB     = float(data['REB'])          # total rebounds
        AST     = float(data['AST'])          # assists
        STL     = float(data['STL'])          # steals
        BLK     = float(data['BLK'])          # blocks
        height  = float(data.get('height_cm', _means['height_cm']))
        weight  = float(data.get('weight_kg', _means['weight_kg']))
    except (KeyError, ValueError) as e:
        return jsonify({'error': f'Missing or invalid field: {e}'}), 400

    # ── Feature engineering (mirrors training pipeline) ────────────────────
    log_MIN = np.log1p(MIN_val)
    log_FTM = np.log1p(FTM)
    log_TOV = np.log1p(TOV)
    log_ORB = np.log1p(ORB)
    log_REB = np.log1p(REB)
    Stocks  = STL + BLK

    row = np.array([[
        log_MIN, PM3, PA3, log_FTM, FTA,
        log_TOV, PF, log_ORB, log_REB, AST,
        Stocks, height, weight
    ]])

    # ── Scale & predict ────────────────────────────────────────────────────
    row_scaled = _scaler.transform(row)
    proba      = _model.predict(row_scaled, verbose=0)[0]   # shape (3,)
    pred_idx   = int(np.argmax(proba))                       # 0, 1, or 2

    label      = LABELS[pred_idx]
    confidence = float(proba[pred_idx])

    return jsonify({
        'class_index':  pred_idx,
        'label_he':     label['he'],
        'label_en':     label['en'],
        'emoji':        label['emoji'],
        'color':        label['color'],
        'confidence':   round(confidence * 100, 1),
        'probabilities': {
            LABELS[i]['en']: round(float(p) * 100, 1)
            for i, p in enumerate(proba)
        }
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("🏀 Basketball Classifier API starting…")
    print(f"   Model path : {MODEL_PATH}")
    print(f"   TensorFlow : {'✓' if TF_AVAILABLE else '✗ not installed'}")
    app.run(debug=True, host='0.0.0.0', port=5000)
