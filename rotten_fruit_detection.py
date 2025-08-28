import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Paths
train_dir = "Dataset/Train"
test_dir = "Dataset/Test"

# Parameters
img_size = (224, 224)
batch_size = 32
epochs = 15
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

# Pretrained MobileNetV2
base_model = MobileNetV2(
    weights="imagenet", 
    include_top=False, 
    input_shape=(224,224,3)
)
base_model.trainable = False  # freeze initial layers

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation="softmax")
])

# Compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=test_gen
)

# Fine-tune (optional)
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-6), loss="categorical_crossentropy", metrics=["accuracy"])
history_ft = model.fit(train_gen, epochs=5, validation_data=test_gen, callbacks = [early_stop])

# Evaluate
loss, acc = model.evaluate(test_gen)
print(f"Test Accuracy: {acc*100:.2f}%")

# Save the Model
model.save("rotten_fruit_model.h5")
