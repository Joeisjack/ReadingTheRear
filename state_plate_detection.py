import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
import numpy as np
import random
import cv2
import os
import sys

# ──────────────────────────────────────────────
# Image utilities
# ──────────────────────────────────────────────

def make_square(img, size=224):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))

    square = np.zeros((size, size, 3), dtype=np.uint8)
    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2
    square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return square

def apply_border(img):
    if random.random() < .3 or True:
        grayness = random.random()
        border_color = (0, 0, 0)  # default to black for grayscale
        if grayness < .1: # colorful
            border_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        else: # grayscale
            gray_value = random.randint(0, 220)
            border_color = (
                gray_value,
                gray_value,
                gray_value
            )
        
        thickness = random.randint(5, 15)

        # Randomly obscure some combination of edges
        random_edge = random.random()
        if random_edge < .2:  # top + bottom
            img[:thickness, :] = border_color
            img[-thickness:, :] = border_color
        elif random_edge < .4:  # left + right
            img[:, :thickness] = border_color
            img[:, -thickness:] = border_color
        else:  # all edges
            img[:thickness, :] = border_color
            img[-thickness:, :] = border_color
            img[:, :thickness] = border_color
            img[:, -thickness:] = border_color

    return img

def random_augment(img, training=True):
    if training:
        

        # Shear
        shear = random.uniform(-0.1, 0.1)
        M_shear = np.array([[1, shear, 0],
            [0, 1,    0]], dtype=np.float32)

        # Rotation
        angle = random.uniform(-5, 5)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M_shear, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # Blur (40% chance)
        if random.random() < .4:
            k = random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (k, k), 0)

        # Brightness / contrast jitter
        alpha = random.uniform(0.8, 1.2)
        beta  = random.uniform(-20, 20)
        img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    return img

def preprocess_image(path, label, size=224, training = True):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if training:
        img = apply_border(img)

    img = make_square(img, size = size)

    img = random_augment(img, training=training)
   

    img = img.astype(np.float32)
    img = preprocess_input(img)
    return img, np.int64(label)

# ──────────────────────────────────────────────
# Dataset builder
# ──────────────────────────────────────────────

def build_dataset(data_dir, size=224, batch_size=32, training=True):
    class_names = sorted(os.listdir(data_dir))
    class_map   = {name: i for i, name in enumerate(class_names)}

    paths, labels = [], []
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(class_dir, fname))
                labels.append(class_map[class_name])

    num_classes = len(class_names)

    def tf_preprocess(path, label):
        img, label = tf.numpy_function(
            lambda p, l: preprocess_image(
                p.decode('utf-8'), l, size=size, training=training
            ),
            [path, label],
            [tf.float32, tf.int64]
        )
        img.set_shape([size, size, 3])
        label.set_shape([])
        label_onehot = tf.one_hot(label, num_classes)
        return img, label_onehot

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=len(paths))
    dataset = (
        dataset
        .map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset, class_map, num_classes

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

def build_state_detector(num_classes=56):
    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,

        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),

        layers.GlobalMaxPooling2D(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation='softmax')
    ])

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy']
    )

    return model

# ──────────────────────────────────────────────
# Sample check
# ──────────────────────────────────────────────

def get_sample(data_dir='dataset/train', size=224):
    class_names = sorted(os.listdir(data_dir))
    some_class  = class_names[0]
    some_file   = os.listdir(os.path.join(data_dir, some_class))[0]
    path        = os.path.join(data_dir, some_class, some_file)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = apply_border(img)
    img = make_square(img, size=size)
    img = random_augment(img, training=True)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("sample.png", img_bgr)
    print(f"Saved sample.png — source: {path}")

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

sys.stdout.isatty = lambda: True

print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

# tf.debugging.set_log_device_placement(True)

train_dataset, class_map, num_classes = build_dataset('dataset/train', training=True)
val_dataset,   _,         _           = build_dataset('dataset/valid', training=False)

get_sample()

model = build_state_detector(num_classes=num_classes)

checkpoint = ModelCheckpoint('best_state_tmodel.keras', monitor = 'val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor = 'val_accuracy', patience = 6, restore_best_weights=True)

model.fit(
    train_dataset,
    epochs = 20,
    validation_data = val_dataset,
    callbacks = [checkpoint, early_stop],
    verbose = 1
)

# ──────────────────────────────────────────────
# Fine-tuning
# ──────────────────────────────────────────────

model = tf.keras.models.load_model('best_state_tmodel.keras')

model.layers[0].trainable = True
for layer in model.layers[0].layers[:-30]:
    layer.trainable = False

loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=loss_fn,
    metrics=['accuracy']
)

early_stop_fine  = EarlyStopping(monitor='val_accuracy', patience=13, restore_best_weights=True, verbose=1)
reduce_lr_fine   = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=8, min_lr=1e-8, verbose=1)
checkpoint_fine  = ModelCheckpoint('best_state_tmodel_fine.keras', monitor='val_accuracy', save_best_only=True)

model.fit(
    train_dataset,
    epochs = 200,
    validation_data = val_dataset,
    callbacks=[early_stop_fine, reduce_lr_fine, checkpoint_fine],
    verbose = 1
)

model.save('StatePlateModel.keras')
print("Final model saved to disk!")

with open('class_indices.json', 'w') as f:
    json.dump(class_map, f)
print("Class indices saved to class_indices.json")