import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset and create the same 80/20 split as Q2 and Q3
breast_cancer_data = load_breast_cancer()
feature_matrix = breast_cancer_data.data
target_vector  = breast_cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, target_vector,
    test_size=0.2,
    random_state=42,
    stratify=target_vector
)

# Standardise features
# Neural networks use gradient descent, which is sensitive to
# feature scale. StandardScaler transforms each feature to have
# mean=0 and standard deviation=1 so no single feature dominates.
# IMPORTANT: fit only on training data to avoid data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # Apply same transform

# Build the Neural Network
# Three hidden layers with ReLU activation, sigmoid output for
# binary classification (outputs a probability between 0 and 1)
tf.random.set_seed(42)

binary_classifier = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1,  activation='sigmoid')   # Binary output unit
], name='BinaryClassifier')

# adam = adaptive learning rate optimiser
# binary_crossentropy = standard loss for binary classification
binary_classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

binary_classifier.summary()

# Train the model
# epoch = one full pass through the entire training dataset
# validation_split reserves 15% of training data to monitor
# generalisation during training without touching the test set
training_history = binary_classifier.fit(
    X_train_scaled, y_train,
    epochs=80,
    batch_size=32,
    validation_split=0.15,
    verbose=0
)

# Report accuracy
train_accuracy = binary_classifier.evaluate(X_train_scaled, y_train, verbose=0)[1]
test_accuracy  = binary_classifier.evaluate(X_test_scaled,  y_test,  verbose=0)[1]

print("--- Neural Network Binary Classifier ---")
print(f"Training Accuracy : {train_accuracy * 100:.2f}%")
print(f"Test Accuracy     : {test_accuracy  * 100:.2f}%")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Loss curve — should decrease smoothly over epochs
axes[0].plot(training_history.history['loss'],     label='Train', color='#E05C5C')
axes[0].plot(training_history.history['val_loss'], label='Val',   color='#5C9BE0', linestyle='--')
axes[0].set_title('Loss Over Epochs', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Binary Cross-Entropy Loss')
axes[0].legend()
axes[0].spines[['top', 'right']].set_visible(False)

# Accuracy curve — should increase and then plateau
axes[1].plot(training_history.history['accuracy'],     label='Train', color='#2ECC71')
axes[1].plot(training_history.history['val_accuracy'], label='Val',   color='#F39C12', linestyle='--')
axes[1].set_title('Accuracy Over Epochs', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].spines[['top', 'right']].set_visible(False)

plt.suptitle('Neural Network Training Curves', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('q4_training_curves.png', dpi=120)
plt.show()

# DISCUSSION:
#
# WHY IS FEATURE SCALING NECESSARY FOR NEURAL NETWORKS?
# Neural networks update weights using gradient descent. When
# features have very different ranges (e.g., one is 0–1 and
# another is 0–1000), gradients along large-scale features
# dominate the updates. This makes the loss surface uneven and
# causes the optimiser to oscillate or converge very slowly.
# Standardising all features to the same scale (mean=0, std=1)
# creates a smoother, more symmetric loss surface, allowing
# the model to learn faster and more stably. Decision trees
# are not affected by scale because they only compare feature
# values against thresholds — scaling does not change rankings.
#
# WHAT IS AN EPOCH?
# One epoch is one complete pass through the entire training
# dataset. During each epoch, the model sees every training
# example once, computes the loss, and adjusts its weights
# via backpropagation. Training for multiple epochs allows the
# model to iteratively improve its weights. Too few epochs
# leads to underfitting; too many can lead to overfitting
# if the model starts memorising training noise.