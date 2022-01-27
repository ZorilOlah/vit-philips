#%%
import pandas as pd
from pathlib import Path
from datasets import load_dataset, load_metric, Features, ClassLabel, Array3D
from PIL import Image
from transformers import ViTModel, ViTConfig, ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
import torch
import numpy as np
from transformers import default_data_collator
from transformers import ViTModel
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

path = Path(__file__).parent

def to_local_drive(excel_file_path, save_location):
    "Takes in Philips excel file (total_schaver_database) and saves all images to local folder."
    database_path = excel_file_path
    total_shaver_database = pd.read_excel(database_path)
    print(total_shaver_database.columns)
    file_locations = total_shaver_database['image_location']
    labels = total_shaver_database['predicted_label']
    names = total_shaver_database['product_image']

    error_counter = 0
    for step, (name, file_path) in enumerate(zip(names, file_locations)):
        try:
            if step%100 == 0:
                print(f'Image number {step}')
            # print(f'name is {name}')
            # print(f'path is {file_path}')
            full_path = save_location + name
            # print(f'full path for saving is : {full_path}')
            with Image.open(str(file_path)) as im:
                im.save(full_path)
        except Exception as e:
            error_counter += 1
            print(f'Error due to : {e}')
    print(f'Total errors : {error_counter}')

to_local_drive(path/'data/total_shaver_database.xlsx', path/'data/image-files/')

# train_ds, test_ds = load_dataset('cifar10', split=['train[:1000]', 'test[:200]']) 
# %%
