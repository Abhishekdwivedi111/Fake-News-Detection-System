
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

feature_names = vectorizer.get_feature_names_out()
importances = model.feature_importances_

feat_df = pd.DataFrame({"feature": feature_names, "importance": importances})
feat_df = feat_df.sort_values(by="importance", ascending=False).head(20)

plt.figure(figsize=(10, 6))
plt.barh(feat_df["feature"], feat_df["importance"], color="steelblue")
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title("Top 20 Most Important Words in Fake News Detection")
plt.tight_layout()

plt.savefig("top_features.png")
print("✅ Saved top feature plot as 'top_features.png'")
