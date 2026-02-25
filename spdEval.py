import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

model = tf.keras.models.load_model('StatePlateModel.keras')

img_path = "eeee.png"

test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size = (160, 320),
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = False
)

print("Generating predictions...")
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis = 1)

y_true = test_generator.classes

class_labels = list(test_generator.class_indices.keys())

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names = class_labels))

img_path

# for state, index in test_generator.class_indices.items():
#     print(f"{index}: {state}")
