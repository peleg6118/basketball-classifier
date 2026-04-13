"""
train_model.py
--------------
Run this script ONCE to train the neural network and save:
  - model/nn_model.keras   (the trained model)
  - model/scaler.pkl       (the StandardScaler)

Usage:
    python train_model.py --data players_stats_by_season_full_details.csv
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout

# ── Config ──────────────────────────────────────────────────────────────────

FEATURES = [
    'log_MIN', '3PM', '3PA',
    'log_FTM', 'FTA', 'log_TOV', 'PF',
    'log_ORB', 'log_REB', 'AST',
    'Stocks', 'height_cm', 'weight_kg'
]
LOG_COLS  = ['PF', 'TOV', 'FGA', 'ORB', 'MIN', 'REB', 'AST', 'FGM', 'FTM']
TARGET    = 'Shooting_Level'
MODEL_DIR = Path(__file__).parent / 'model'
MODEL_DIR.mkdir(exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in LOG_COLS:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
    df['Stocks'] = df['BLK'] + df['STL']
    df['PPG']    = (df['PTS'] / df['GP']).round(1)
    df['FG_pct'] = (df['FGM'] / df['FGA'] * 100).round(1)
    conds = [
        (df['PPG'] > 25) | ((df['FG_pct'] > 40) & (df['PPG'] > 20)),
        ((df['FG_pct'] < 40) & (df['PPG'] < 15)) | ((df['FG_pct'] > 40) & (df['PPG'] < 10))
    ]
    df[TARGET] = np.select(conds, ['1', '3'], default='2')
    df[TARGET]  = pd.to_numeric(df[TARGET], errors='coerce')
    return df


def build_model(input_dim: int, n_classes: int) -> Sequential:
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ── Main ─────────────────────────────────────────────────────────────────────

def main(csv_path: str):
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    df = preprocess(df)

    data = df[FEATURES + [TARGET]].dropna()
    X    = data[FEATURES]
    y    = data[TARGET]

    print(f"Dataset shape: {X.shape}  |  Class distribution:\n{y.value_counts().sort_index()}")

    # Scale → SMOTEENN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote_enn = SMOTEENN(random_state=42)
    X_res, y_res = smote_enn.fit_resample(X_scaled, y)
    print(f"After SMOTEENN: {X_res.shape}  |  {pd.Series(y_res).value_counts().sort_index().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    n_classes = len(np.unique(y_res))
    y_train_cat = tf.keras.utils.to_categorical(y_train - 1, num_classes=n_classes)
    y_test_cat  = tf.keras.utils.to_categorical(y_test  - 1, num_classes=n_classes)

    model = build_model(X_train.shape[1], n_classes)
    model.summary()

    print("\nTraining neural network (50 epochs)…")
    model.fit(X_train, y_train_cat,
              epochs=50, batch_size=32,
              validation_data=(X_test, y_test_cat),
              verbose=1)

    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}")

    # Save
    model_path  = MODEL_DIR / 'nn_model.keras'
    scaler_path = MODEL_DIR / 'scaler.pkl'
    model.save(str(model_path))
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Save dataset means for height/weight imputation
    means = {'height_cm': float(df['height_cm'].mean()), 'weight_kg': float(df['weight_kg'].mean())}
    with open(MODEL_DIR / 'means.pkl', 'wb') as f:
        pickle.dump(means, f)

    print(f"\nSaved model  → {model_path}")
    print(f"Saved scaler → {scaler_path}")
    print("Done ✓")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='players_stats_by_season_full_details.csv',
                        help='Path to the CSV dataset')
    args = parser.parse_args()
    main(args.data)
