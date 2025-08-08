import os
import cv2
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.data_loader import DATASET_DIR, MAX_AGE

def load_and_preprocess(file_name, img_size=128):
    file_path = os.path.join(DATASET_DIR, file_name)
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    return img.astype(np.float32)

def extract_labels_from_filename(file_path):
    parts = file_path.split('_')
    age = int(parts[0]) / MAX_AGE
    gender = int(parts[1])
    return gender, age

def create_datagen(augment=False):
    if augment:
        return ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
    return ImageDataGenerator(rescale=1./255)

def multi_output_generator(file_list, datagen, batch_size=32, img_size=128, shuffle=True):
    while True:
        if shuffle:
            random.shuffle(file_list)
        for i in range(0, len(file_list), batch_size):
            batch_files = file_list[i:i+batch_size]
            batch_images, gender_labels, age_labels = [], [], []
            for file in batch_files:
                img = load_and_preprocess(file, img_size)
                gender, age = extract_labels_from_filename(file)
                batch_images.append(img)
                gender_labels.append(gender)
                age_labels.append(age)
            batch_images = np.array(batch_images)
            batch_images_aug = next(datagen.flow(batch_images, batch_size=len(batch_images), shuffle=False))
            yield batch_images_aug, {
                'gender_output': np.array(gender_labels),
                'age_output': np.array(age_labels, dtype=np.float32)
            }
