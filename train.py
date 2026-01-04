import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 48
BATCH = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\AMIT KUMAR\OneDrive\Desktop\AI_Powered_Music_Generator_Based_On_User_Mood\Data\train",
    image_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH,
    shuffle=True)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\AMIT KUMAR\OneDrive\Desktop\AI_Powered_Music_Generator_Based_On_User_Mood\Data\test",
    image_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH,
    shuffle=True)

norm = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (norm(x), y))
test_ds = test_ds.map(lambda x, y: (norm(x), y))

model = models.Sequential([
    layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(48,48,1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, validation_data=test_ds, epochs=25)

model.save("emotion_model.h5")
print("Model saved as emotion_model.h5")