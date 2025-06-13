import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path

# Define absolute paths relative to backend/
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "clips_dataset.csv"
MODEL_PATH = BASE_DIR / "app/model/xgboost_virality_model.pkl"

# Load dataset and model
df = pd.read_csv(CSV_PATH)
model = joblib.load(MODEL_PATH)

# Select only unlabeled rows
unlabeled_df = df[df['label'].isna()]

if not unlabeled_df.empty:
    features = unlabeled_df[["llm_score", "sentiment", "length_sec", "keyword_hit"]]
    predicted_labels = model.predict(features)
    df.loc[unlabeled_df.index, "label"] = predicted_labels.astype(int)
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ Labeled {len(predicted_labels)} new clips!")
else:
    print("✅ No unlabeled clips found.")
