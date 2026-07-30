import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import MobileNetV2
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIRECTORY =r'C:\Users\rishi\Downloads\DATASET\TRAIN' 
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15 
if not os.path.isdir(DATA_DIRECTORY):
    print(f"Error: Data directory '{DATA_DIRECTORY}' not found.")
    print("Please create the folder and put your 5 pose folders inside it.")
    exit()


datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    DATA_DIRECTORY,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    DATA_DIRECTORY,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False 
)

num_classes = train_generator.num_classes
class_names = list(train_generator.class_indices.keys())

print("-" * 50)
print(f"Found {num_classes} classes: {class_names}")
print("-" * 50)



base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,         
    weights='imagenet'         
)

base_model.trainable = False

model = Sequential([
    base_model,                     
    Flatten(),                      
    Dense(512, activation='relu'),  
    Dropout(0.5),                  
    Dense(num_classes, activation='softmax') 
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

model.summary()


print("\nStarting model training...")

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS
)

MODEL_PATH = 'yoga_pose_detector.h5'
model.save(MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
