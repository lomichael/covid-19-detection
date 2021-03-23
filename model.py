
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

model = Sequential()

# Convolutional layers
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation="relu", input_shape=(224, 224, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))

# Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Randomly drops neurons from being selected for activation to prevent overfitting
model.add(Dropout(0.25))

# Convolutional layer
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))

# Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout to prevent overfitting
model.add(Dropout(0.25))

# Flatten layer
model.add(Flatten())

# Densely fully connected layer
model.add(Dense(64, activation="relu"))

# Dropout to prevent overfitting
model.add(Dropout(0.5))

# Output node
model.add(Dense(1, activation="sigmoid"))

# Binary loss function for binary classification
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])

# Printing the model
model.summary()

# Creating data generators that will generate variations of the images
train_datagen = image.ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = image.ImageDataGenerator(rescale=1./255)

# Loading the dataset
train_generator = train_datagen.flow_from_directory(
    'data/train-set', target_size=(224, 224), batch_size=32, class_mode="binary")
val_generator = test_datagen.flow_from_directory(
    'data/test-set', target_size=(224, 224), batch_size=32, class_mode="binary")

hist = model.fit(train_generator, epochs=6,
                 validation_data=val_generator, validation_steps=2)
