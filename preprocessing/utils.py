from csv import excel
from pyexpat import features
from typing import Any
from pathlib import Path
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from transformers import ViTModel, ViTConfig, ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
from datasets import load_dataset, load_metric, Features, ClassLabel, Array3D, Value
from transformers import default_data_collator
import numpy as np
from PIL import Image
import numpy as np
from pathlib import Path
from transformers.modeling_outputs import SequenceClassifierOutput
from ViT2 import ViTForImageClassification2
import os
from io import BytesIO  
import pandas as pd

def available_images(excel_file_path, image_folder):
    "checks which of the images in de total_shaver_database are actually accesible on device" 
    data = pd.read_excel(excel_file_path) 
    new_names, new_labels = [], []
    names = list(data['product_image'])
    labels = list(data['predicted_label'])
    errors = 0
    wrong_shape = 0
    missing_data = 0
    for name, label in zip(names, labels):
        try:
            if os.path.isfile(image_folder + name):
                image = Image.open(image_folder+name)
                image = image.resize((600,600), Image.ANTIALIAS)
                image = np.array(image, dtype=np.uint8)
                image = np.moveaxis(image, source=-1, destination=0)
                if image.shape == (3, 600, 600):
                    new_names.append(image_folder + name)
                    new_labels.append(label)
                else:
                    wrong_shape += 1
            else:
                missing_data +=1
        except:
            errors += 1
            continue
    print(f'available images did not work for {errors} files, {wrong_shape} images were the wrong shape, and {missing_data} images were in the excel sheet but not in the image-folder')
    return new_names, new_labels

def split_dataset(dataset, ratio):
    splits = dataset.train_test_split(test_size=ratio)
    train_ds = splits['train']
    test_ds = splits['test'] 
    return train_ds, test_ds


