import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Load the MNIST dataset (images of handwritten digits)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the images to a range of 0 to 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple neural network model
model = Sequential([
    layers.Flatten(input_shape=(28, 28)),    # Flatten images into 1D vectors
    layers.Dense(128, activation='relu'),    # Fully connected layer with 128 neurons
    layers.Dropout(0.2),                     # Dropout layer to prevent overfitting
    layers.Dense(10, activation='softmax')   # Output layer with 10 units (digits 0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the training data
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
