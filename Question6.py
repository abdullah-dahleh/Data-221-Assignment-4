import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import fashion_mnist

# Human-readable names for the 10 clothing categories
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Load Fashion MNIST
# 60,000 training images and 10,000 test images, 28x28 grayscale
(X_train_images, y_train_labels), (X_test_images, y_test_labels) = fashion_mnist.load_data()

print(f"Training images shape : {X_train_images.shape}")
print(f"Test images shape     : {X_test_images.shape}")

# Normalise pixel values to [0, 1]
# Raw pixels range 0–255. Dividing by 255 keeps all values small,
# which helps gradient descent converge faster and more stably.
X_train_normalised = X_train_images.astype('float32') / 255.0
X_test_normalised  = X_test_images.astype('float32')  / 255.0

# Add the channel dimension
# Keras Conv2D expects input shape (height, width, channels).
# Grayscale has 1 channel, so we reshape from (28,28) to (28,28,1).
X_train_reshaped = X_train_normalised.reshape(-1, 28, 28, 1)
X_test_reshaped  = X_test_normalised.reshape(-1, 28, 28, 1)

print(f"\nAfter preprocessing:")
print(f"  X_train_reshaped : {X_train_reshaped.shape}")
print(f"  X_test_reshaped  : {X_test_reshaped.shape}")

# Show sample images
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_train_reshaped[i, :, :, 0], cmap='gray')
    ax.set_title(CLASS_NAMES[y_train_labels[i]], fontsize=8)
    ax.axis('off')
plt.suptitle('Fashion MNIST — Sample Training Images', fontweight='bold')
plt.tight_layout()
plt.savefig('q6_sample_images.png', dpi=120)
plt.show()

# Build the CNN
# Block 1: detect basic edges and textures (32 filters)
# Block 2: detect more complex patterns like straps or soles (64 filters)
# Block 3: detect high-level shapes specific to clothing (128 filters)
# MaxPooling halves the spatial dimensions, reducing computation
# Flatten converts the 2D feature maps into a 1D vector for Dense layers
# Dropout randomly disables 40% of neurons during training to reduce overfitting
tf.random.set_seed(42)

fashion_cnn = Sequential([
    Conv2D(32,  (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64,  (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')   # 10 output neurons, one per class
], name='FashionCNN')

# sparse_categorical_crossentropy is used when labels are integers (not one-hot)
fashion_cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

fashion_cnn.summary()

# Train for at least 15 epochs
cnn_training_history = fashion_cnn.fit(
    X_train_reshaped, y_train_labels,
    epochs=15,
    batch_size=128,
    validation_split=0.1,   # 10% of training data used for validation
    verbose=1
)

# Evaluate on the test set
test_loss, test_accuracy = fashion_cnn.evaluate(X_test_reshaped, y_test_labels, verbose=0)
print(f"\n--- CNN on Fashion MNIST ---")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(cnn_training_history.history['loss'],     label='Train', color='#E05C5C')
axes[0].plot(cnn_training_history.history['val_loss'], label='Val',   color='#5C9BE0', linestyle='--')
axes[0].set_title('CNN Loss Over Epochs', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].spines[['top', 'right']].set_visible(False)

axes[1].plot(cnn_training_history.history['accuracy'],     label='Train', color='#2ECC71')
axes[1].plot(cnn_training_history.history['val_accuracy'], label='Val',   color='#F39C12', linestyle='--')
axes[1].set_title('CNN Accuracy Over Epochs', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].spines[['top', 'right']].set_visible(False)

plt.suptitle('CNN Training Curves — Fashion MNIST', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('q6_training_curves.png', dpi=120)
plt.show()

# Save the trained model so Q7 can load it instead of retraining
fashion_cnn.save('q6_fashion_cnn_model.keras')
print("Model saved to q6_fashion_cnn_model.keras")

# DISCUSSION:
#
# WHY ARE CNNs BETTER THAN FULLY CONNECTED NETWORKS FOR IMAGES?
# A fully connected (Dense) layer treats every pixel as an
# independent input, losing all spatial relationships between
# neighbouring pixels and requiring an enormous number of
# parameters. CNNs use small sliding filters (kernels) that
# share weights across the entire image, exploiting two key
# properties: local connectivity (nearby pixels are correlated)
# and translation invariance (a feature like an edge is useful
# regardless of where it appears in the image). This results
# in far fewer parameters, better generalisation, and much
# stronger performance on image data.
#
# WHAT IS THE CONVOLUTION LAYER LEARNING HERE?
# Early Conv2D layers learn low-level visual features: edges,
# curves, and fabric textures. Middle layers combine these into
# mid-level patterns like straps, soles, sleeves, and buckles.
# The final dense layers use these patterns to make the final
# classification decision for each of the 10 clothing types.