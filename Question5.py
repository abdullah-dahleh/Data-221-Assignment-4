import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Rebuild both models so this file runs independently

breast_cancer_data = load_breast_cancer()
feature_matrix = breast_cancer_data.data
target_vector  = breast_cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, target_vector,
    test_size=0.2,
    random_state=42,
    stratify=target_vector
)

# Constrained Decision Tree (same settings as Q3)
constrained_tree = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
constrained_tree.fit(X_train, y_train)

# Neural Network (same settings as Q4)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

tf.random.set_seed(42)
binary_classifier = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1,  activation='sigmoid')
])
binary_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
binary_classifier.fit(X_train_scaled, y_train, epochs=80, batch_size=32,
                      validation_split=0.15, verbose=0)

# Generate predictions
tree_predictions = constrained_tree.predict(X_test)

# Convert sigmoid probabilities to binary labels (threshold = 0.5)
nn_probabilities  = binary_classifier.predict(X_test_scaled)
nn_predictions    = (nn_probabilities >= 0.5).astype(int).ravel()

# Compute confusion matrices
tree_confusion_matrix = confusion_matrix(y_test, tree_predictions)
nn_confusion_matrix   = confusion_matrix(y_test, nn_predictions)

# Plot both confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, cm, title in zip(
    axes,
    [tree_confusion_matrix, nn_confusion_matrix],
    ['Constrained Decision Tree', 'Neural Network']
):
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=breast_cancer_data.target_names
    )
    display.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontweight='bold', fontsize=11)

plt.suptitle('Confusion Matrices — Test Set', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('q5_confusion_matrices.png', dpi=120)
plt.show()

# Print key metrics derived from each confusion matrix
def print_derived_metrics(confusion_mat, model_name):
    tn, fp, fn, tp = confusion_mat.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity
    accuracy  = (tp + tn) / confusion_mat.sum()
    print(f"\n  {model_name}")
    print(f"    Accuracy  : {accuracy  * 100:.2f}%")
    print(f"    Precision : {precision * 100:.2f}%")
    print(f"    Recall    : {recall    * 100:.2f}%")
    print(f"    TP={tp}  FP={fp}  TN={tn}  FN={fn}")

print("--- Model Comparison ---")
print_derived_metrics(tree_confusion_matrix, "Constrained Decision Tree")
print_derived_metrics(nn_confusion_matrix,   "Neural Network")

# DISCUSSION:
#
# WHICH MODEL IS PREFERRED FOR THIS TASK?
# For cancer diagnosis, recall on the malignant class is the
# most critical metric — a false negative (missed cancer) is
# far more dangerous than a false positive. If the neural
# network achieves higher recall, it is the safer clinical
# choice. However, the decision tree's interpretability is a
# significant practical advantage: clinicians can follow its
# decision rules and explain predictions to patients.
#
# DECISION TREE:
#   Advantage  — Fully interpretable. Every prediction can be
#                traced back to a human-readable rule, which
#                is critical in regulated medical settings.
#   Limitation — High variance. Small changes in training data
#                can produce a completely different tree.
#
# NEURAL NETWORK:
#   Advantage  — Can model complex, non-linear feature
#                interactions that a shallow tree may miss,
#                often achieving higher accuracy and recall.
#   Limitation — Acts as a black box. It is difficult to
#                explain why a specific prediction was made,
#                which is a barrier in high-stakes healthcare.