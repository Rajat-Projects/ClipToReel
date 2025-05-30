# Any clips that are still unlabeled (label column is empty) should be automatically classified using your trained XGBoost model.

import pandas as pd
import xgboost as xgb
import joblib

# Load dataset and model
df = pd.read_csv("clips_dataset.csv")
model = joblib.load("virality_xgb_model.pkl")

# Select only unlabeled rows
unlabeled_df = df[df['label'].isna()]

if not unlabeled_df.empty:
    features = unlabeled_df[["llm_score", "sentiment", "length_sec", "keyword_hit"]]
    predicted_labels = model.predict(features)
    df.loc[unlabeled_df.index, "label"] = predicted_labels.astype(int)
    df.to_csv("clips_dataset.csv", index=False)
    print(f"✅ Labeled {len(predicted_labels)} new clips!")
else:
    print("✅ No unlabeled clips found.")
