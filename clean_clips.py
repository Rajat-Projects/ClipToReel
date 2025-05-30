import pandas as pd

# Load the dataset
df = pd.read_csv("clips_dataset.csv")

# Drop duplicates based on the clip path
df_unique = df.drop_duplicates(subset=["clip_path"])

# Save the cleaned dataset
df_unique.to_csv("clips_dataset_cleaned.csv", index=False)
