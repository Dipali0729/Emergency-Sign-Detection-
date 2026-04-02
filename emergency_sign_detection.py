import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 📁 Dataset path
dataset_path = "EmergencySignDataset/Image_Data"

# 🔥 Data Augmentation (IMPORTANT)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5]
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ✅ Print classes
print("Classes:", train_generator.class_indices)

# 🧠 Model (Improved)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # 🔥 Prevent overfitting
    Dense(len(train_generator.class_indices), activation='softmax')
])

# ⚙️ Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 🚀 Train
print("🚀 Training started...")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15
)

# 💾 Save
model.save("emergency_model.h5")
print("✅ Model saved as emergency_model.h5")