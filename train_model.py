import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# --------------------------------
# Dataset Path
# --------------------------------
dataset_path = r"D:\Clg Project\Alzheimer Disease\Data"

# --------------------------------
# Image Preprocessing
# --------------------------------
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

# --------------------------------
# Training Data
# --------------------------------
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# --------------------------------
# Validation Data
# --------------------------------
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# --------------------------------
# CNN Model
# --------------------------------
model = models.Sequential([

    layers.Input(shape=(128, 128, 3)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dropout(0.5),

    layers.Dense(4, activation='softmax')
])

# --------------------------------
# Compile Model
# --------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --------------------------------
# Train Model
# --------------------------------
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# --------------------------------
# Save Model
# --------------------------------
model.save("alzheimer_model.keras")

print("✅ Model Saved Successfully")
