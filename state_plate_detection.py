import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
import numpy as np

def build_state_detector(num_classes = 56):
    base_model = MobileNetV2(input_shape = (128, 224, 3), include_top = False, weights = 'imagenet')

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation = 'relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation = 'softmax')
    ])

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

model = build_state_detector()

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 5,
    width_shift_range = 0.05,
    height_shift_range = 0.05,
    shear_range = 0.05,
    brightness_range = [0.7, 1.3]
)

val_datagen = ImageDataGenerator(
    rescale = 1./255
)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size = (128, 224),
    batch_size = 32,
    class_mode = 'categorical',
)

validation_generator = val_datagen.flow_from_directory(
    'dataset/valid',
    target_size = (128, 224),
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = False
)

class_map = train_generator.class_indices
custom_weights = {i: 1.0 for i in range(56)}

# States that are harder to decern from one another
hard_states = {
    'RHODE ISLAND': 1.5,
    'SOUTH CAROLINA': 1.4,
    'VIRGINIA': 1.1,
    'WEST VIRGINIA': 1.1,
    'ALABAMA': 1.08,
    'TEXAS': 1.05,
}

for state, weight in hard_states.items():
    if state in class_map:
        custom_weights[class_map[state]] = weight
    else:
        print(f"Warning: State '{state}' not found in class indices. Skipping weight adjustment.")

checkpoint = ModelCheckpoint('best_state_tmodel.keras', monitor='val_accuracy', save_best_only=True)

model.fit(train_generator, epochs = 20, validation_data = validation_generator, callbacks = [checkpoint])

model = tf.keras.models.load_model('best_state_tmodel.keras')

model.layers[0].trainable = True
for layer in model.layers[0].layers[:-10]:
    layer.trainable = False

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5), 
    loss = 'categorical_crossentropy', 
    metrics = ['accuracy']
)

early_stop_fine = EarlyStopping(monitor = 'val_accuracy', patience = 20, restore_best_weights = True, verbose = 1)
reduce_lr_fine = ReduceLROnPlateau(monitor='val_loss', factor = 0.2, patience = 5, min_lr = 1e-8, verbose = 1)
checkpoint_fine = ModelCheckpoint('best_state_tmodel_fine.keras', monitor = 'val_accuracy', save_best_only = True)

# Pre adding weights
# model.fit(
#     train_generator, 
#     epochs = 3, 
#     validation_data = validation_generator, 
#     callbacks = [early_stop_fine, reduce_lr_fine, checkpoint_fine]
# )

model.fit(
    train_generator, 
    epochs = 100, 
    validation_data = validation_generator, 
    callbacks = [early_stop_fine, reduce_lr_fine, checkpoint_fine]
    # class_weight = custom_weights
)

model.save('StatePlateModel.keras') 
print("Final model saved to disk!")

with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
print("Class indices saved to class_indices.json")