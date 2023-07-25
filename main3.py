import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


# Load eyeglasses and non-eyeglasses image data
def load_data(data_dir, img_size=(64, 64)):
    X = []
    y = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image = cv2.resize(image, img_size)  # Resize image to a fixed size
            X.append(image)
            y.append(
                int(label)
            )  # Assuming '0' for no eyeglasses and '1' for eyeglasses

    return np.array(X), np.array(y)


# Define the CNN architecture
def create_eyeglasses_detection_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))

    return model


# Load data and preprocess it
data_dir = "path/to/your/dataset"  # Replace this with the path to your dataset
X, y = load_data(data_dir)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize pixel values to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Specify the input shape of your images (height, width, channels)
input_shape = X_train[0].shape

# Create the model
model = create_eyeglasses_detection_model(input_shape)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy: {:.2f}%".format(accuracy * 100))
