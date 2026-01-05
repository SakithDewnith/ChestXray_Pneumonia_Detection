import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model 

# Paths
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'pneumonia_model.h5')

def train_engine():
    # 1. Setup Data Generators (With Augmentation)
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR, target_size=(150, 150), batch_size=32,
        class_mode='binary', subset='training'
    )
    val_generator = train_datagen.flow_from_directory(
        DATA_DIR, target_size=(150, 150), batch_size=32,
        class_mode='binary', subset='validation'
    )
    
    # 2. Train
    print("Starting Training...")
    model = build_model()
    model.fit(train_generator, epochs=5, validation_data=val_generator)
    
    # 3. Save the Brain
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_engine()
