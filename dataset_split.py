import os
from sklearn.model_selection import train_test_split
import shutil

TRAIN_DIR = "dataset_train"
VAL_DIR = "dataset_val"
DATA_DIR = "Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid) with Symptoms and SMOTE Validation"
TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

classes = os.listdir(DATA_DIR)
for cls in classes:
    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)

for cls in classes:
    cls_path = os.path.join(DATA_DIR, cls)
    images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    train_imgs, val_imgs = train_test_split(images, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Move images
    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(TRAIN_DIR, cls, img))

    for img in val_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(VAL_DIR, cls, img))

print("Train/test split completed!")