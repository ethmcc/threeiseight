import tensorflow as tf


def train_digit_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess data with explicit type casting
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Define the simplified model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # Compile with default adam optimizer
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train the model with simplified parameters
    model.fit(
        x_train,
        y_train,
        epochs=8,
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Loss:", loss)
    print("Accuracy:", accuracy)

    # Save model in native Keras format
    model.save("digit_model.keras")


if __name__ == "__main__":
    train_digit_model()
