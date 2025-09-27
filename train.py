import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import MobileNetV2
import os
from PIL import ImageFile
# This line tells the image library to ignore the "truncated file" error
ImageFile.LOAD_TRUNCATED_IMAGES = True 

# The rest of your imports follow...
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# ...

# --- 1. Configuration ---
# !!! RENAME this to the folder containing your 5 pose subfolders !!!
DATA_DIRECTORY =r'C:\Users\rishi\Downloads\DATASET\TRAIN' 
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15 # You may need to increase this number for better results

# Check if the data directory exists
if not os.path.isdir(DATA_DIRECTORY):
    print(f"Error: Data directory '{DATA_DIRECTORY}' not found.")
    print("Please create the folder and put your 5 pose folders inside it.")
    exit()

# --- 2. Data Preparation and Augmentation ---

# ImageDataGenerator handles loading, rescaling, and data augmentation
# Rescale normalizes pixel values from 0-255 to 0-1
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, # Reserve 20% of data for validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load training data
train_generator = datagen.flow_from_directory(
    DATA_DIRECTORY,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Load validation data
validation_generator = datagen.flow_from_directory(
    DATA_DIRECTORY,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False # Do not shuffle validation data
)

num_classes = train_generator.num_classes
class_names = list(train_generator.class_indices.keys())

print("-" * 50)
print(f"Found {num_classes} classes: {class_names}")
print("-" * 50)


# --- 3. Model Definition (Transfer Learning) ---

# Load MobileNetV2 base model, pre-trained on ImageNet
# We use Transfer Learning to leverage a model already trained on millions of images.
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,         # Exclude the default final classification layer
    weights='imagenet'         # Use pre-trained weights
)

# Freeze the base layers so we only train the new classification head
base_model.trainable = False

# Build the new model head
model = Sequential([
    base_model,                     # MobileNetV2 feature extractor
    Flatten(),                      # Flatten the 3D feature map into a 1D vector
    Dense(512, activation='relu'),  # A new dense layer for learning specific yoga features
    Dropout(0.5),                   # Dropout to reduce overfitting
    # The final layer: num_classes will automatically be 5, matching your data
    Dense(num_classes, activation='softmax') 
])

# --- 4. Compile the Model ---
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', # Appropriate loss for multi-class classification
    metrics=['accuracy']
)

model.summary()


# --- 5. Train the Model ---
print("\nStarting model training...")

# Fit the model using the data generators
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS
)

# --- 6. Save Model ---
MODEL_PATH = 'yoga_pose_detector.h5'
model.save(MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")