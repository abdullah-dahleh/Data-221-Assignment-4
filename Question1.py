import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Load the Breast Cancer Wisconsin dataset from scikit-learn
breast_cancer_data = load_breast_cancer()

# Build the feature matrix (X) and target vector (y)
feature_matrix = breast_cancer_data.data    # Shape: (569, 30)
target_vector  = breast_cancer_data.target  # 0 = malignant, 1 = benign

# Report the shape of X and y
print("Shape of feature matrix X:", feature_matrix.shape)
print("Shape of target vector  y:", target_vector.shape)

# Count how many samples belong to each class
class_labels, class_counts = np.unique(target_vector, return_counts=True)

print("\nSample count per class:")
for label, count in zip(class_labels, class_counts):
    class_name = breast_cancer_data.target_names[label]
    print(f"  Class {label} ({class_name}): {count} samples")

# Calculate the ratio between classes to assess balance
imbalance_ratio = class_counts[1] / class_counts[0]
print(f"\nBenign-to-Malignant ratio: {imbalance_ratio:.2f}")

# ── Bar chart of class distribution ──────────────────────────
fig, ax = plt.subplots(figsize=(5, 3))
bar_colors = ['#E05C5C', '#5C9BE0']  # Red = malignant, Blue = benign
bars = ax.bar(breast_cancer_data.target_names, class_counts,
              color=bar_colors, edgecolor='white', width=0.5)

# Label each bar with its count
ax.bar_label(bars, fmt='%d', padding=4, fontsize=11)
ax.set_title('Class Distribution — Breast Cancer Dataset', fontweight='bold')
ax.set_ylabel('Number of Samples')
ax.set_ylim(0, max(class_counts) * 1.15)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('q1_class_distribution.png', dpi=120)
plt.show()

# ─────────────────────────────────────────────────────────────
# DISCUSSION (written in comments as required):
#
# IS THE DATASET BALANCED OR IMBALANCED?
# The dataset contains 212 malignant and 357 benign samples.
# This is a mild imbalance (~1.68:1 ratio favouring benign).
# It is not extreme, but it is worth considering during modelling.
#
# WHY DOES CLASS BALANCE MATTER?
# A model trained on imbalanced data can achieve high accuracy
# simply by always predicting the majority class (benign here).
# In a medical context, this is dangerous — missing a malignant
# tumour (a false negative) has life-threatening consequences.
# Balanced classes, or strategies like class weighting or
# recall-focused metrics, ensure the model works well for BOTH
# classes, not just the one that appears most often.
# ─────────────────────────────────────────────────────────────