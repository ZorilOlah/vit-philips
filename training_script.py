# %%
from csv import excel
from typing import Any
from ShaverDataset import ShaverDataset
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

path = Path(__file__).parent

image_folder = str(path) + '/data/image_folder/'
excelsheet = str(path) + '/data/total_shaver_database.xlsx'

def available_images(excel_file_path, image_folder):
    "checks which of the images in de total_shaver_database are actually accesible on device" 
    data = pd.read_excel(excel_file_path) 
    new_names, new_labels = [], []
    names = list(data['product_image'])
    labels = list(data['predicted_label'])
    for name, label in zip(names, labels):
        if os.path.isfile(image_folder + name):
            image = Image.open(image_folder+name)
            image = image.resize((600,600), Image.ANTIALIAS)
            image = np.array(image, dtype=np.uint8)
            image = np.moveaxis(image, source=-1, destination=0)
            if image.shape == (3, 600, 600):
                new_names.append(image_folder + name)
                new_labels.append(label)
    return new_names, new_labels

names_list, labels_list = available_images(excelsheet, image_folder)

unique_labels = list(set(labels_list))

int_labels_list = [unique_labels.index(label) for label in labels_list]

available_dataset = pd.DataFrame({'img': names_list, 'label': int_labels_list})
available_dataset.to_csv(str(path) + '/data/available_dataset.csv', index = False)

dataset = load_dataset('csv', data_files=str(path) + '/data/available_dataset.csv', delimiter = ',')['train']

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

features = Features({
    'label': ClassLabel(names=unique_labels),
    'img': Value(dtype = "string", id = None),
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
})

def preprocess_images(examples):
    paths = examples['img']
    images = [path_to_image(image_path = path) for path in paths]
    # print(images)
    # print(type(images))
    images = [np.array(image, dtype=np.uint8) for image in images]
    images = [np.moveaxis(image, source=-1, destination=0) for image in images]
    for image in images:
        print(image.shape)
    # print(images[0])
    # print(images[0].shape)
    inputs = feature_extractor(images=images)
    examples['pixel_values'] = inputs['pixel_values']
    return examples


def path_to_image(image_path):
    image = Image.open(image_path)
    image = image.resize((600,600), Image.ANTIALIAS)
    return image

preprocessed_train_ds = dataset.map(preprocess_images, batched=True, features=features)

# %%
