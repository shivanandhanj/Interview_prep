import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Debugging: Print the contents of the dataset directory
dataset_path = 'dataset'
print(f"Contents of '{dataset_path}': {os.listdir(dataset_path)}")

for class_dir in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_dir)
    print(f"Contents of '{class_path}': {os.listdir(class_path)}")

# Set up data generators with a smaller validation split
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)  # Reduced validation split

# Create train generator
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
print(f"Found {train_generator.samples} training images.")

# Create validation generator
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
print(f"Found {validation_generator.samples} validation images.")

# Proceed only if generators have samples
if train_generator.samples > 0 and validation_generator.samples > 0:
    # Build the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(150, 150, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator
    )

    # Save the model
    model.save('touch_head_model.h5')
else:
    print("No samples found. Please check the dataset directory and paths.")
