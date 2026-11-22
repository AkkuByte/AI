import numpy as np
import tensorflow as tf
from PIL import Image

# 1. Load the built-in 28x28 grayscale handwritten digits dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Scale pixel values from [0, 255] to [0.0, 1.0] for faster training
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Define a simple 3-layer neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),                       # Unrolls 28x28 grid into 784 pixels
    tf.keras.layers.Dense(128, activation="relu"),   # Hidden layer learns edges and curves
    tf.keras.layers.Dense(10, activation="softmax")  # Outputs probabilities for digits 0-9
])

# 4. Compile and train (reaches ~97-98% accuracy in a few seconds)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 5. Predict a number from your own 28x28 black-and-white image
def predict_digit(image_path):
    img = Image.open(image_path).convert("L").resize((28, 28))
    img_array = np.array(img) / 255.0

    # MNIST expects a white digit on a black background.
    # If your image has a black digit on a white background, uncomment the next line:
    # img_array = 1.0 - img_array

    img_batch = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 28, 28)
    predictions = model.predict(img_batch, verbose=0)

    digit = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    print(f"Predicted Number: {digit} ({confidence:.1f}% confidence)")

# Example usage:
# predict_digit("my_number.png")
# 5. Save the trained model so you don't have to retrain every time
model.save("digit_model.keras")
print("Model saved as digit_model.keras")

# 6. Predict a number from your own 28x28 black-and-white image
def predict_digit(image_path):
    img = Image.open(image_path).convert("L").resize((28, 28))
    img_array = np.array(img) / 255.0

    # Auto-invert if the background is white (average brightness > 0.5)
    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array

    img_batch = np.expand_dims(img_array, axis=0)  # Shape: (1, 28, 28)
    predictions = model.predict(img_batch, verbose=0)

    digit = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    print(f"Predicted Number: {digit} ({confidence:.1f}% confidence)")

# Quick sanity check using the very first image from the MNIST test set
Image.fromarray((x_test[0] * 255).astype(np.uint8)).save("sample_test.png")
print(f"Actual label for sample_test.png: {y_test[0]}")
predict_digit("sample_test.png")

# To test your own drawing, save a 28x28 PNG in the same folder and uncomment:
# predict_digit("my_number.png")
