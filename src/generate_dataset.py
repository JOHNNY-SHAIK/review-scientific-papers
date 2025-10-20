import pandas as pd
import random
import os
import json

# --- Step 1: Define templates for each sentiment ---
positive_contexts = [
    "This method significantly improves performance compared to previous work.",
    "The results demonstrate superior accuracy and robustness.",
    "Our findings confirm the effectiveness of this approach.",
    "This technique provides remarkable improvements over baselines.",
    "The proposed algorithm achieves outstanding results on benchmark datasets."
]

neutral_contexts = [
    "This study follows the standard methodology described by Smith et al.",
    "The results are comparable to those found in previous research.",
    "Our model uses a similar architecture to the prior approach.",
    "This work adopts techniques commonly used in machine learning studies.",
    "The findings align with earlier observations in this field."
]

negative_contexts = [
    "However, the method fails to generalize well to unseen data.",
    "The results show limitations in scalability and robustness.",
    "This approach underperforms compared to more recent models.",
    "Our analysis reveals several weaknesses in the proposed framework.",
    "The findings suggest the method is less effective than expected."
]

# --- Step 2: Generate 405 fake records (135 each) ---
records = []
for i in range(135):
    records.append({
        "paper_id": f"P{i+1:03}",
        "citation_context": random.choice(positive_contexts),
        "label": "Positive"
    })
for i in range(135, 270):
    records.append({
        "paper_id": f"P{i+1:03}",
        "citation_context": random.choice(neutral_contexts),
        "label": "Neutral"
    })
for i in range(270, 405):
    records.append({
        "paper_id": f"P{i+1:03}",
        "citation_context": random.choice(negative_contexts),
        "label": "Negative"
    })

# --- Step 3: Convert to DataFrame ---
df = pd.DataFrame(records)
print("✅ Dataset created successfully!")
print(df.head(), "\n")
print("Shape of dataset:", df.shape)

# --- Step 4: Save to /data folder ---
os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", "mldataset_full.json")
df.to_json(output_path, orient="records", indent=4)
print(f"✅ Saved dataset to: {output_path}")
