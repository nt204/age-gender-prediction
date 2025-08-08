import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATASET_DIR = "/kaggle/input/utkface-new/UTKFace"
MAX_AGE = 116

def show_random_image():
    file_list = [f for f in os.listdir(DATASET_DIR) if f.endswith(".jpg")]
    random_file = random.choice(file_list)
    parts = random_file.split("_")
    age = int(parts[0])
    gender = "Male" if int(parts[1]) == 0 else "Female"

    img_path = os.path.join(DATASET_DIR, random_file)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.title(f"Age: {age}, Gender: {gender}")
    plt.axis('off')
    plt.show()

def gender_distribution():
    male_count = female_count = 0
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith(".jpg"):
            try:
                gender = int(filename.split("_")[1])
                if gender == 0:
                    male_count += 1
                else:
                    female_count += 1
            except:
                continue
    plt.bar(['Male', 'Female'], [male_count, female_count], color=['blue', 'pink'])
    plt.title("Gender Distribution in UTKFace")
    plt.ylabel("Số lượng ảnh")
    plt.show()

def age_distribution():
    ages = []
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith(".jpg"):
            try:
                age = int(filename.split("_")[0])
                ages.append(age)
            except:
                continue
    plt.figure(figsize=(10, 5))
    plt.hist(ages, bins=range(0, 101, 5), color='skyblue', edgecolor='black')
    plt.title("Phân bố tuổi trong UTKFace")
    plt.xlabel("Tuổi")
    plt.ylabel("Số lượng ảnh")
    plt.grid(True)
    plt.xticks(range(0, 101, 5))
    plt.show()

def split_dataset():
    file_list = [f for f in os.listdir(DATASET_DIR) if f.endswith(".jpg")]
    random.shuffle(file_list)
    train_files, temp_files = train_test_split(file_list, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    return train_files, val_files, test_files
