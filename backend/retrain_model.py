import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path

# === CONFIGURATION ===
DATA_PATH = Path("backend/clips_dataset.csv")
MODEL_PATH = Path("backend/app/model/virality_model.pkl")

def retrain_xgboost_model():
    print("📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Drop rows with missing labels
    df = df.dropna(subset=["label"])
    if df.empty:
        print("⚠️ No labeled data available. Aborting training.")
        return

    # Prepare features and labels
    X = df[["llm_score", "sentiment", "length_sec", "keyword_hit"]]
    y = df["label"].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    print("🎯 Training XGBoost classifier...")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Evaluate
    print("\n📊 Classification Report:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Model retrained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    retrain_xgboost_model()
