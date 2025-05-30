import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load labeled dataset
df = pd.read_csv("clips_dataset.csv")

# Keep only labeled samples
df = df[df["label"].isin([0, 1])]

# Features and target
X = df[["llm_score", "sentiment", "length_sec", "keyword_hit"]]
y = df["label"]

# Split into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost classifier
model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "virality_model.pkl")
print("✅ Trained model saved as 'virality_model.pkl'")

# Evaluate model
y_pred = model.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
