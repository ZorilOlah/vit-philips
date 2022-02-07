# %%
from csv import excel
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
from preprocessing.utils import available_images, split_dataset

path = Path(__file__).parent

image_folder = str(path) + '/data/image-files'
# image_folder = str(path) + '/data/image_folder/'
excelsheet = str(path) + '/data/total_shaver_database.xlsx'

names_list, labels_list = available_images(excelsheet, image_folder)
unique_labels = list(set(labels_list))
int_labels_list = [unique_labels.index(label) for label in labels_list]
available_dataset = pd.DataFrame({'img': names_list, 'label': int_labels_list})
available_dataset.to_csv(str(path) + '/data/available_dataset.csv', index = False)

dataset = load_dataset('csv', data_files=str(path) + '/data/available_dataset.csv', delimiter = ',', split = ['train'])[0]
train_val, test = split_dataset(dataset = dataset, ratio = 0.1)
train, val = split_dataset(dataset = train_val, ratio = 0.2)


feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

features = Features({
    'label': ClassLabel(names=unique_labels),
    'img': Value(dtype = "string", id = None),
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
})
def path_to_image(image_path):
    image = Image.open(image_path)
    image = image.resize((600,600), Image.ANTIALIAS)
    return image

def preprocess_images(examples):
    paths = examples['img']
    images = [path_to_image(image_path = path) for path in paths]
    # print(images)
    # print(type(images))
    images = [np.array(image, dtype=np.uint8) for image in images]
    images = [np.moveaxis(image, source=-1, destination=0) for image in images]
    # for image in images:
    #     print(image.shape)
    # print(images[0])
    # print(images[0].shape)
    inputs = feature_extractor(images=images)
    examples['pixel_values'] = inputs['pixel_values']
    return examples

preprocessed_train_ds = train.map(preprocess_images, batched=True, features=features)
preprocessed_val_ds = val.map(preprocess_images, batched=True, features=features)
preprocessed_test_ds = test.map(preprocess_images, batched=True, features=features)

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
# model = ViTForImageClassification2()

metric = load_metric("accuracy")
data_collator = default_data_collator

model.train()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

args = TrainingArguments(
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=False,
    metric_for_best_model='accuracy',
    logging_dir='./logs',
    output_dir = './results'
)

trainer = Trainer(
    model,
    args,
    train_dataset = preprocessed_train_ds,
    eval_dataset = preprocessed_val_ds,
    data_collator = data_collator,
    compute_metrics = compute_metrics,
)

trainer.train()

# %%
