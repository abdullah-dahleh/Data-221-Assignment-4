import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load dataset and split (same seed as Q2 for consistency)
breast_cancer_data = load_breast_cancer()
feature_matrix = breast_cancer_data.data
target_vector  = breast_cancer_data.target
feature_names  = breast_cancer_data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, target_vector,
    test_size=0.2,
    random_state=42,
    stratify=target_vector
)

# Train a constrained Decision Tree
# max_depth=4        > tree can make at most 4 levels of splits
# min_samples_split  > a node needs at least 10 samples to split
# min_samples_leaf   > every leaf must contain at least 5 samples
# These constraints prevent the tree from memorising noise
constrained_tree = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
constrained_tree.fit(X_train, y_train)

# Evaluate accuracy
train_accuracy = accuracy_score(y_train, constrained_tree.predict(X_train))
test_accuracy  = accuracy_score(y_test,  constrained_tree.predict(X_test))

print("--- Constrained Decision Tree ---")
print(f"Training Accuracy : {train_accuracy * 100:.2f}%")
print(f"Test Accuracy     : {test_accuracy  * 100:.2f}%")
print(f"Tree Depth        : {constrained_tree.get_depth()}")
print(f"Number of Leaves  : {constrained_tree.get_n_leaves()}")

# Top 5 most important features
# Feature importance = total entropy reduction due to that feature
feature_importances = constrained_tree.feature_importances_
top5_indices = np.argsort(feature_importances)[::-1][:5]

print("\nTop 5 Most Important Features:")
print(f"  {'Rank':<5} {'Feature Name':<35} {'Importance':>10}")
print("  " + "-" * 52)
for rank, idx in enumerate(top5_indices, start=1):
    print(f"  {rank:<5} {feature_names[idx]:<35} {feature_importances[idx]:>10.4f}")

# Bar chart: feature importances
fig, ax = plt.subplots(figsize=(7, 3.5))
top5_names  = [feature_names[i] for i in top5_indices][::-1]
top5_values = feature_importances[top5_indices][::-1]

ax.barh(top5_names, top5_values, color='#2E86AB', edgecolor='white')
ax.set_xlabel('Feature Importance Score')
ax.set_title('Top 5 Feature Importances — Constrained Decision Tree', fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('q3_feature_importance.png', dpi=120)
plt.show()

# Tree diagram
# Visualises the actual decision rules the tree learned
fig, ax = plt.subplots(figsize=(18, 7))
plot_tree(
    constrained_tree,
    feature_names=feature_names,
    class_names=breast_cancer_data.target_names,
    filled=True,    # Colour nodes by majority class
    rounded=True,
    fontsize=8,
    ax=ax
)
ax.set_title('Constrained Decision Tree (max_depth=4)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('q3_tree_diagram.png', dpi=100)
plt.show()

# DISCUSSION:
#
# HOW DO CONSTRAINTS REDUCE OVERFITTING?
# Without limits, a tree grows until all leaves are pure,
# memorising every training sample including noise. Adding
# max_depth caps how many rules it can chain together.
# min_samples_split stops the tree from creating splits based
# on tiny groups of samples that may just be outliers.
# min_samples_leaf ensures every rule covers enough examples
# to be statistically reliable. Together these constraints
# force the model to learn broader, more general patterns
# rather than quirks of the training data — reducing the
# gap between training and test accuracy.
#
# HOW DOES FEATURE IMPORTANCE HELP INTERPRETABILITY?
# Feature importance tells us which measurements were most
# useful for separating malignant from benign tumours. A domain
# expert (e.g., an oncologist) can check whether these features
# are clinically meaningful, which builds trust in the model.
# It can also guide future data collection by highlighting which
# measurements matter most. This transparency is one of the
# biggest advantages decision trees have over black-box models.