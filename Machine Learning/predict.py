import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "digit_model.keras"

# 1. Load saved model (or train and save it once if not found)
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Training model for the first time...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}\n")


# 2. Preprocess any image into MNIST format (28x28 white digit on black background)
def prepare_image(image_path):
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)

    # Auto-invert if the outer border/background is bright (e.g. black ink on white paper)
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if np.mean(border) > 127:
        arr = 255.0 - arr

    # Remove faint gray background shadows
    arr[arr < 50] = 0.0

    # Crop tightly around the digit and center it inside a 28x28 frame
    coords = np.argwhere(arr > 0)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = Image.fromarray(arr[y0:y1, x0:x1].astype(np.uint8))

        w, h = cropped.size
        scale = 20.0 / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

        canvas = Image.new("L", (28, 28), 0)
        canvas.paste(resized, ((28 - new_w) // 2, (28 - new_h) // 2))
        final_arr = np.array(canvas, dtype=np.float32) / 255.0
    else:
        final_arr = np.zeros((28, 28), dtype=np.float32)

    return final_arr


# 3. Predict the digit
def predict_digit(image_path):
    processed = prepare_image(image_path)
    img_batch = np.expand_dims(processed, axis=0)  # Shape: (1, 28, 28)

    predictions = model.predict(img_batch, verbose=0)[0]
    top_3 = np.argsort(predictions)[::-1][:3]

    print(f"\nImage: {image_path}")
    print(f"Predicted Number: {top_3[0]} ({predictions[top_3[0]] * 100:.1f}% confidence)")
    print("Top 3 guesses:")
    for digit in top_3:
        print(f"  Digit {digit}: {predictions[digit] * 100:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py /path/to/your_image.png")
    else:
        predict_digit(sys.argv[1])
