# model/train_model.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

# --- CONFIG ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # project root (DeepX-ChestXRay)
DATA_DIR = os.path.join(ROOT, "dataset", "chest_xray", "train")        # use train folder and split internally
MODEL_DIR = os.path.join(ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
INITIAL_EPOCHS = 8
FINETUNE_EPOCHS = 6
TOTAL_EPOCHS = INITIAL_EPOCHS + FINETUNE_EPOCHS
VALIDATION_SPLIT = 0.15

# --- DATA GENERATORS (use validation_split to get a meaningful val set) ---
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=VALIDATION_SPLIT
)

val_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=VALIDATION_SPLIT
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=SEED
)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=SEED
)

# --- MODEL (Transfer Learning with DenseNet121) ---
base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
base_model.trainable = False  # freeze for initial training

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(5e-4))(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --- CALLBACKS ---
checkpoint_path = os.path.join(MODEL_DIR, "cnn_model.h5")
checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-7)

# --- INITIAL TRAINING (head only) ---
print(">>> START: initial training (head only)")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS,
    callbacks=[checkpoint, earlystop, reduce_lr],
    verbose=1
)

# Save a backup in TF native format
model.save(os.path.join(MODEL_DIR, "cnn_model_backup.keras"))

# --- FINETUNING: unfreeze top layers of base model ---
print(">>> START: fine-tuning")
base_model.trainable = True

# Freeze lower layers, unfreeze top N layers for fine-tuning
# Adjust this number if your DenseNet has different layer count
fine_tune_at = int(len(base_model.layers) * 0.7)

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

# lower LR for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=TOTAL_EPOCHS,
    initial_epoch=history1.epoch[-1] + 1 if history1.epoch else INITIAL_EPOCHS,
    callbacks=[checkpoint, earlystop, reduce_lr],
    verbose=1
)

# Final save (overwrite best .h5 if checkpoint saved it earlier, but save native format too)
model.save(checkpoint_path)  # HDF5 legacy (works for app)
model.save(os.path.join(MODEL_DIR, "cnn_model_final.keras"))

print("Training complete. Best model saved to:", checkpoint_path)
