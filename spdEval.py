import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model('StatePlateModel.keras')

img_path = "eeee.png"

test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size = (128, 224),
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



# for state, index in test_generator.class_indices.items():
#     print(f"{index}: {state}")
