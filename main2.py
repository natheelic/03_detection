# Build an eyeglasses detection CNN model

# Data Collection

# Import the required libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Step 1: Prepare the Dataset
dataset_path = "dataset"


# Define the eyeglasses CNN architecture
def create_eyeglasses_detection_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


# Specify the input shape of your images (height, width, channels)
input_shape = (150, 150, 3)

# Create the model
model = create_eyeglasses_detection_model(input_shape)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# train the model
model.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=32,
    validation_data=(test_images, test_labels),
)


# Step 2: Load the dataset eyeglasses detection from kaggle
import kaggle

kaggle.api.authenticate()
kaggle.api.dataset_download_files(
    "ashishjangra27/face-mask-12k-images-dataset", path=dataset_path, unzip=True
)

# Step 3: Load and preprocess the dataset
# Define image size and batch size
img_width, img_height = 150, 150
batch_size = 32

# Step 4: Use ImageDataGenerator for data augmentation and normalization
train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

# Step 5: Use flow_from_directory() to generate batches of image data from the train and validation folders
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
)
# Load and preprocess the validation data
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
)

# Step 6: Build an eyeglasses detection CNN model
# Build a CNN model
model = models.Sequential()
model.add(
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 3))
)
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))


model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.5))

# Step 7: Compile the model
model.add(layers.Dense(1, activation="sigmoid"))
model.compile(
    loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"]
)

# Step 8: Train the model
epochs = 10
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
)

# Step 9: Save the model
model.save("eyeglasses_detector.h5")

# Step 10: Evaluate the model
# Plot the training and validation accuracy and loss at each epoch
import matplotlib.pyplot as plt

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()

# Step 11: Test the model
# Load the model
from tensorflow.keras.models import load_model

model = load_model("eyeglasses_detector.h5")

# Load the test dataset
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate_generator(test_generator, batch_size)
print("test loss, test acc:", results)

# Generate predictions (probabilities) on the test data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict_generator(test_generator, steps=3)
print("predictions shape:", predictions.shape)

# Step 12: Visualize the predictions
# Visualize the predictions
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

# Get the filenames from the generator
fnames = test_generator.filenames

# Get the ground truth from generator
ground_truth = test_generator.classes

# Get the label to class mapping from the generator
label2index = test_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v, k) for k, v in label2index.items())

# Get the predictions from the model using the generator
predictions = model.predict_generator(
    test_generator, steps=test_generator.samples // batch_size + 1, verbose=1
)
predicted_classes = np.argmax(predictions, axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors), test_generator.samples))

# Let's see which we predicted correctly and which not
correct_predictions = np.where(predicted_classes == ground_truth)[0]
print(
    "No of correct predictions = {}/{}".format(
        len(correct_predictions), test_generator.samples
    )
)


# Adapted from
# https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
def to_one_hot(y, n_class):
    return np.eye(n_class)[y.reshape(-1)]


# Convert predictions classes to one hot vectors
Y_pred_classes = to_one_hot(predicted_classes, 2)
# Convert validation observations to one hot vectors

Y_true = to_one_hot(ground_truth, 2)

from sklearn.metrics import classification_report

print(classification_report(Y_true, Y_pred_classes, target_names=["0", "1"]))

# Get the confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(ground_truth, predicted_classes)
print(cm)

# Visualize the confusion matrix
import seaborn as sns

plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True)
plt.show()

# Get the classification report
from sklearn.metrics import classification_report

report = classification_report(ground_truth, predicted_classes, target_names=["0", "1"])
print(report)

# Visualize the classification report
import pandas as pd

report_data = []
lines = report.split("\n")
for line in lines[2:-3]:
    row = {}
    row_data = line.split("      ")
    row["class"] = row_data[1]
    row["precision"] = float(row_data[2])
    row["recall"] = float(row_data[3])
    row["f1_score"] = float(row_data[4])
    row["support"] = float(row_data[5])
    report_data.append(row)
df = pd.DataFrame.from_dict(report_data)
df.to_csv("classification_report.csv", index=False)

# Step 14: Test the model on a new image
# Load the image
img = image.load_img("test_image.jpg", target_size=(img_width, img_height))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)

# Predict the image
output = model.predict(img)
if output[0][0] > output[0][1]:
    print("No Eyeglasses Detected")
else:
    print("Eyeglasses Detected")

# Step 15: Convert the model to TensorFlow Lite
# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open("eyeglasses_detector.tflite", "wb") as f:
    f.write(tflite_model)

# Step 16: Test the TensorFlow Lite model
# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="eyeglasses_detector.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the image
img = image.load_img("test_image.jpg", target_size=(img_width, img_height))
img = np.asarray(img)
img = np.expand_dims(img, axis=0)

# Test the TensorFlow Lite model on the image
input_shape = input_details[0]["shape"]
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]["index"], img)

interpreter.invoke()

# Extract output data from the interpreter
output_data = interpreter.get_tensor(output_details[0]["index"])
if output_data[0][0] > output_data[0][1]:
    print("No Eyeglasses Detected")
else:
    print("Eyeglasses Detected")

# Step 17: Convert the TensorFlow Lite model to a C source file
# Convert the TensorFlow Lite model to a C source file
import subprocess

subprocess.run(
    [
        "xxd",
        "-i",
        "eyeglasses_detector.tflite",
        "eyeglasses_detector_model_data.cc",
    ]
)

# Step 18: Create a C++ application to run inference on a new image
# Create a C++ application to run inference on a new image
import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# To capture video from webcam.
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for x, y, w, h in faces:
        # Crop the face
        roi = img[y : y + h, x : x + w]
        # Resize the face
        roi = cv2.resize(roi, (img_width, img_height))
        # Save the face
        cv2.imwrite("test_image.jpg", roi)
        # Display
        cv2.imshow("img", roi)
        # Exit condition
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
# Release the VideoCapture object
cap.release()

# Step 19: Build the C++ application
# Build the C++ application
subprocess.run(["cmake", "-S", ".", "-B", "build"])
subprocess.run(["cmake", "--build", "build"])
