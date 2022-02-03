import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from transformers import ViTModel, ViTConfig, ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
from datasets import load_dataset, load_metric, Features, ClassLabel, Array3D
from transformers import default_data_collator
import numpy as np
from PIL import Image
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class ShaverDataset(Dataset):

    def __init__(self, image_dir, excel_file_path, transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.data = pd.read_excel(excel_file_path)
        self.labels = list(self.data['predicted_label'])
        self.index_labels = list(set(self.labels))
        self.names = list(self.data['product_image'])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_name)  
        try:
            image = Image.open(image_path)
            label_index = self.names.index(image_name)            
            label_str = self.labels[label_index]
            label_int = self.index_labels.index(label_str) 
            if self.transform:
                image = self.transform(image)
            return {'pixel_values' : image['pixel_values'], 'label' : label_int}
        except:
            print(f"Could not find label for : {image_name}")