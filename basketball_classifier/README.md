# 🏀 Basketball Player Classifier

> רשת נוירונים לסיווג שחקני כדורסל — דיוק 91%

## מבנה הפרויקט

```
basketball_classifier/
├── app.py              ← Flask backend (API)
├── train_model.py      ← אימון המודל ושמירתו
├── requirements.txt    ← תלויות Python
├── README.md
├── model/              ← נוצר אוטומטית לאחר האימון
│   ├── nn_model.keras
│   ├── scaler.pkl
│   └── means.pkl
└── templates/
    └── index.html      ← ממשק המשתמש
```

---

## התקנה והפעלה

### שלב 1 – התקן תלויות
```bash
pip install -r requirements.txt
```

### שלב 2 – אמן את המודל
מקם את הקובץ `players_stats_by_season_full_details.csv` בתיקיית הפרויקט, ואז הרץ:
```bash
python train_model.py --data players_stats_by_season_full_details.csv
```
הסקריפט יאמן את הרשת (50 epochs) וישמור את המודל ב-`model/`.

### שלב 3 – הפעל את השרת
```bash
python app.py
```

פתח דפדפן בכתובת: **http://localhost:5000**

---

## API Endpoints

| Method | Path       | Description                     |
|--------|------------|---------------------------------|
| GET    | `/`        | ממשק המשתמש (HTML)              |
| POST   | `/predict` | קבלת נתוני שחקן → סיווג         |
| GET    | `/health`  | בדיקת תקינות השרת               |

### דוגמת POST /predict
```json
{
  "MIN": 32.0,
  "3PM": 2.5,
  "3PA": 6.1,
  "FTM": 4.2,
  "FTA": 5.0,
  "TOV": 2.1,
  "PF":  2.4,
  "ORB": 0.8,
  "REB": 5.3,
  "AST": 6.7,
  "STL": 1.5,
  "BLK": 0.3,
  "height_cm": 193,
  "weight_kg": 91
}
```

### תשובה לדוגמה
```json
{
  "class_index": 0,
  "label_he": "שחקן טוב",
  "label_en": "Good Player",
  "emoji": "🌟",
  "color": "#F5A623",
  "confidence": 94.2,
  "probabilities": {
    "Good Player": 94.2,
    "Average Player": 4.1,
    "Below Average": 1.7
  }
}
```

---

## לוגיקת הסיווג

| Shooting Level | תנאי                                                        |
|----------------|-------------------------------------------------------------|
| 🌟 טוב (1)     | PPG > 25 **או** (FG% > 40% **וגם** PPG > 20)              |
| ⚡ ממוצע (2)   | ברירת מחדל                                                  |
| 📉 לא טוב (3)  | (FG% < 40% **וגם** PPG < 15) **או** (FG% > 40% **וגם** PPG < 10) |

---

## ארכיטקטורת המודל
```
Input(13) → Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.3) → Dense(3, softmax)
```
אימון: SMOTEENN + StandardScaler, 50 epochs, Adam optimizer

---

*פרויקט: עידו ופלג*
