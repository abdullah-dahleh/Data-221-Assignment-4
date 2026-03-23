from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
breast_cancer_data = load_breast_cancer()
feature_matrix = breast_cancer_data.data
target_vector  = breast_cancer_data.target

# 80/20 stratified split
# stratify=y ensures both splits keep the same class ratio
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, target_vector,
    test_size=0.2,
    random_state=42,
    stratify=target_vector
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")

# Train a Decision Tree using entropy as the split criterion
# Entropy measures disorder in a node - the tree picks splits
# that reduce entropy the most (maximum information gain)
entropy_decision_tree = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42       # Fixed seed for reproducibility
)
entropy_decision_tree.fit(X_train, y_train)

# Evaluate on training and test sets
train_predictions = entropy_decision_tree.predict(X_train)
test_predictions  = entropy_decision_tree.predict(X_test)

training_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy     = accuracy_score(y_test,  test_predictions)

print("\n--- Decision Tree (Entropy, No Constraints) ---")
print(f"Training Accuracy : {training_accuracy * 100:.2f}%")
print(f"Test Accuracy     : {test_accuracy     * 100:.2f}%")
print(f"Tree Depth        : {entropy_decision_tree.get_depth()}")
print(f"Number of Leaves  : {entropy_decision_tree.get_n_leaves()}")

# DISCUSSION:
#
# WHAT IS ENTROPY IN A DECISION TREE?
# Entropy measures the impurity (uncertainty) of a group of
# samples. For a binary class with probability p, it is:
#   H = -p * log2(p) - (1-p) * log2(1-p)
# A node where all samples belong to one class has entropy = 0
# (perfectly pure). Equal class split gives entropy = 1 (maximum
# uncertainty). At each node, the algorithm picks the feature
# and threshold that produce the largest Information Gain:
#   Information Gain = parent entropy - weighted child entropy
# This greedily reduces uncertainty at every step.
#
# DOES THE RESULT SUGGEST OVERFITTING?
# A training accuracy of 100% with a noticeably lower test
# accuracy is a textbook sign of overfitting. Without any depth
# constraint, the tree keeps splitting until every leaf contains
# only one class — essentially memorising the training data,
# including any noise. It has not learned to generalise.
# This will be addressed in Q3 by adding constraints.