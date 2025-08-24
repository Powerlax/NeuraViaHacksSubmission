wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/n9zp473wfw-2.zip
unzip n9zp473wfw-2.zip
del "Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid) with Symptoms and SMOTE Validation/Duplicate.ipynb"
del "Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid) with Symptoms and SMOTE Validation/Eye_Diseases_Classifications_Pre_Processing_Dataset.ipynb"
python dataset_split.py
python model_training.py