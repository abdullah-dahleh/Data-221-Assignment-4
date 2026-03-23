import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Human-readable names for the 10 clothing categories
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Load and preprocess Fashion MNIST (same steps as Q6)
(_, _), (X_test_images, y_test_labels) = fashion_mnist.load_data()

X_test_normalised = X_test_images.astype('float32') / 255.0
X_test_reshaped   = X_test_normalised.reshape(-1, 28, 28, 1)

# Load the trained CNN saved by Q6
fashion_cnn = load_model('q6_fashion_cnn_model.keras')
print("Model loaded successfully.")

# Generate predictions on the full test set
# predict() returns a probability for each of the 10 classes
predicted_probabilities = fashion_cnn.predict(X_test_reshaped, verbose=0)

# Pick the class with the highest probability as the final prediction
predicted_class_indices = np.argmax(predicted_probabilities, axis=1)

# Compute and display the confusion matrix
cnn_confusion_matrix = confusion_matrix(y_test_labels, predicted_class_indices)

fig, ax = plt.subplots(figsize=(10, 8))
display = ConfusionMatrixDisplay(
    confusion_matrix=cnn_confusion_matrix,
    display_labels=CLASS_NAMES
)
display.plot(ax=ax, colorbar=True, cmap='Blues', xticks_rotation=45)
ax.set_title('CNN Confusion Matrix — Fashion MNIST Test Set', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('q7_confusion_matrix.png', dpi=120)
plt.show()

# Print the overall test accuracy
overall_accuracy = (predicted_class_indices == y_test_labels).mean()
print(f"\nOverall Test Accuracy : {overall_accuracy * 100:.2f}%")

# Print per-class accuracy
print("\nPer-class Accuracy:")
print(f"  {'Class':<15} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
print("  " + "-" * 44)
for class_index in range(10):
    class_mask       = y_test_labels == class_index
    correct_count    = (predicted_class_indices[class_mask] == class_index).sum()
    total_count      = class_mask.sum()
    class_accuracy   = correct_count / total_count * 100
    print(f"  {CLASS_NAMES[class_index]:<15} {correct_count:>8} {total_count:>8} {class_accuracy:>9.1f}%")

# Find all misclassified samples
misclassified_indices = np.where(predicted_class_indices != y_test_labels)[0]
total_errors = len(misclassified_indices)
error_rate   = total_errors / len(y_test_labels) * 100

print(f"\nTotal misclassified : {total_errors} / {len(y_test_labels)}")
print(f"Misclassification rate : {error_rate:.2f}%")

# Visualise at least 3 misclassified images
# We show 6 for a richer analysis; each displays true vs predicted label
np.random.seed(42)
selected_error_indices = np.random.choice(misclassified_indices, size=6, replace=False)

fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for ax, error_idx in zip(axes.ravel(), selected_error_indices):
    true_class_name      = CLASS_NAMES[y_test_labels[error_idx]]
    predicted_class_name = CLASS_NAMES[predicted_class_indices[error_idx]]
    prediction_confidence = predicted_probabilities[error_idx, predicted_class_indices[error_idx]] * 100

    ax.imshow(X_test_reshaped[error_idx, :, :, 0], cmap='gray')
    ax.set_title(
        f"True  : {true_class_name}\n"
        f"Pred  : {predicted_class_name} ({prediction_confidence:.1f}%)",
        fontsize=9, color='#C0392B', fontweight='bold'
    )
    ax.axis('off')

plt.suptitle('CNN Misclassified Samples — Fashion MNIST', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('q7_misclassified_images.png', dpi=120)
plt.show()

# DISCUSSION:
#
# OBSERVED PATTERN IN MISCLASSIFICATIONS:
# The most common errors occur between visually similar
# categories. 'Shirt' is frequently confused with 'T-shirt/top',
# 'Pullover', and 'Coat' because all four are torso-covering
# garments with similar silhouettes at 28x28 resolution.
# Likewise, footwear categories (Sandal, Sneaker, Ankle boot)
# are confused with each other because they share similar
# bottom-heavy shapes. These errors cluster in the confusion
# matrix — off-diagonal values are concentrated in semantically
# related pairs, not scattered randomly. This tells us the model
# has learned meaningful structure but struggles with fine-
# grained discrimination between similar-looking categories.
#
# ONE METHOD TO IMPROVE CNN PERFORMANCE:
# Data augmentation — applying random transformations to training
# images such as horizontal flips, small rotations, zoom, and
# brightness changes. This artificially expands the training set
# and prevents the CNN from memorising specific orientations or
# lighting conditions. It is especially helpful for the confused
# classes like Shirt vs T-shirt, where seeing the garment from
# slightly different angles can help the model learn more robust
# distinguishing features. In Keras this is done easily with
# tf.keras.layers.RandomFlip and tf.keras.layers.RandomRotation,
# adding no extra labelling cost.