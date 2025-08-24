import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2S
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# -------------------
# Parameters
# -------------------
IMG_SIZE = (384, 384)   # EfficientNetV2-S input size
BATCH_SIZE = 32
EPOCHS = 30
TRAIN_DIR = "dataset_train"
VAL_DIR = "dataset_val"

# -------------------
# Data Augmentation
# -------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

# -------------------
# Class Weights
# -------------------
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))

# -------------------
# Model
# -------------------
base_model = EfficientNetV2S(weights="imagenet", include_top=False, input_shape=IMG_SIZE+(3,))
base_model.trainable = False  # freeze initially

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(train_generator.num_classes, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=outputs)

# -------------------
# Compile
# -------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------
# Callbacks
# -------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
]

# -------------------
# Step 1: Train frozen
# -------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=200,
    class_weight=class_weights,
    callbacks=callbacks
)

# -------------------
# Step 2: Fine-tune (unfreeze last layers)
# -------------------
base_model.trainable = True
for layer in base_model.layers[:-100]:  # freeze all but last 100 layers
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)
