# %%
from pathlib import Path
from tensorboard import summary
# from transformers import ViTModel, ViTConfig, ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
# from datasets import load_dataset, load_metric, Features, ClassLabel, Array3D, Value
# from transformers import default_data_collator
from pathlib import Path
# from ViT2 import ViTForImageClassification2
import pandas as pd
from preprocessing.utils import available_images, split_dataset, preprocess_images, path_to_image, available_images_cropped
from training_parameters.utils import hyperparamters_from_dict, compute_metrics, parameters_to_trainingarguments
from typing import List, Set, Dict, Tuple, Optional
import shortuuid
from results.utils.utils import to_pickle, load_pickle, merge_dicts, best_results_from_log, get_results_dataframe_if_exists

path = Path(__file__).parent

# image_folder = str(path) + '/data/image-files/'
# # image_folder = str(path) + '/data/image_folder/'
# image_folder = str(path) + '/data/total_shaver_database_cropped_subimages/'
# excelsheet = str(path) + '/data/total_shaver_database.xlsx'

# names_list, labels_list = available_images_cropped(excel_file_path = excelsheet, image_folder = image_folder, resize = 600)
# print(f'length of names_list : {len(names_list)}')
# unique_labels = list(set(labels_list))
# print(unique_labels)
# print(f'amount of labels is {len(unique_labels)}')
# print(labels_list[0:10])
# labels = pd.DataFrame(labels_list)
# print(labels.value_counts())

# int_labels_list = [unique_labels.index(label) for label in labels_list]
# available_dataset = pd.DataFrame({'img': names_list, 'label': int_labels_list})
# available_dataset.to_csv(str(path) + '/data/available_dataset_cropped.csv', index = False)
# %%

image_folder = str(path) + '/data/image-files/'
# image_folder = str(path) + '/data/image_folder/'
excelsheet = str(path) + '/data/total_shaver_database.xlsx'

names_list, labels_list = available_images(excel_file_path = excelsheet, image_folder = image_folder, resize = 600)
print(f'length names list : {len(names_list)}')
unique_labels = list(set(labels_list))
print(unique_labels)
print(f'amount of labels is {len(unique_labels)}')

labels = pd.DataFrame(labels_list)
print(labels.value_counts())