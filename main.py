import tensorflow as tf
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Step 1: Prepare the Dataset
dataset_path = "dataset"

# Step 2: Load the dataset eyeglasses detection from kaggle
# import kaggle
# kaggle.api.authenticate()
# kaggle.api.dataset_download_files('ashishjangra27/face-mask-12k-images-dataset', path=dataset_path, unzip=True)


# Define image size and batch size
img_width, img_height = 150, 150
batch_size = 32

# Use ImageDataGenerator for data augmentation and normalization
train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

# Use flow_from_directory() to generate batches of image data from the train and validation folders
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

# Build an eyeglasses detection CNN model


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
model.add(layers.Dense(1, activation="sigmoid"))

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Display the model summary
model.summary()
# save the model
model.save("model.h5")

# Train the model
# Define the number of training and validation steps per epoch
nb_train_samples = 5216
nb_validation_samples = 624
epochs = 10
train_steps_per_epoch = nb_train_samples // batch_size
validation_steps_per_epoch = nb_validation_samples // batch_size

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps_per_epoch,
)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(validation_generator)
print("Test accuracy: {:.2f}%".format(accuracy * 100))

# Plot the training and validation accuracy and loss at each epoch
# import matplotlib.pyplot as plt
# %matplotlib inline

# Plot training and validation accuracy per epoch
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Training and validation accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# Plot training and validation loss per epoch
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Training and validation loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# Make predictions on the test data
# import numpy as np
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model

# Load the saved model
# model = load_model('model.h5')

# Load the image file, resizing it to 150 x 150 pixels
# img = image.load_img('test_image.png', target_size=(150, 150))

# Convert the image to a numpy array
# x = image.img_to_array(img)

# Add a fourth dimension to the image (since Keras expects a list of images)
# x = np.expand_dims(x, axis=0)

# Normalize the image
# x = x / 255.0

# Run the image through the deep neural network to make a prediction
# prediction = model.predict(x)

# print(prediction)

# if prediction == 0:
#     print("The X-ray scan shows the patient has a Pneumonia.")
# else:
#     print("The X-ray scan shows the patient is healthy.")

# Save the model
# model.save('model.h5')
