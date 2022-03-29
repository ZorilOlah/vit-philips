# %%
from pathlib import Path
from tensorboard import summary
from transformers import ViTModel, ViTConfig, ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
from datasets import load_dataset, load_metric, Features, ClassLabel, Array3D, Value
from transformers import default_data_collator
from pathlib import Path
from ViT2 import ViTForImageClassification2
import pandas as pd
from preprocessing.utils import available_images, split_dataset, preprocess_images, path_to_image
from training_parameters.utils import hyperparamters_from_dict, compute_metrics, parameters_to_trainingarguments
from typing import List, Set, Dict, Tuple, Optional
import shortuuid
from results.utils.utils import to_pickle, load_pickle, merge_dicts, best_results_from_log, get_results_dataframe_if_exists

path = Path(__file__).parent

# image_folder = str(path) + '/data/image-files/'
image_folder = str(path) + '/data/image_folder/'
excelsheet = str(path) + '/data/total_shaver_database.xlsx'

names_list, labels_list = available_images(excel_file_path = excelsheet, image_folder = image_folder, resize = 600)
unique_labels = list(set(labels_list))
int_labels_list = [unique_labels.index(label) for label in labels_list]
available_dataset = pd.DataFrame({'img': names_list, 'label': int_labels_list})
available_dataset.to_csv(str(path) + '/data/available_dataset.csv', index = False)

dataset = load_dataset('csv', data_files=str(path) + '/data/available_dataset.csv', delimiter = ',', split = ['train'])[0]
train_val, test = split_dataset(dataset = dataset, ratio = 0.1)
train, val = split_dataset(dataset = train_val, ratio = 0.2)

features = Features({
    'label': ClassLabel(names=unique_labels),
    'img': Value(dtype = "string", id = None),
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
})

preprocessed_train_ds = train.map(preprocess_images, batched=True, features=features)
preprocessed_val_ds = val.map(preprocess_images, batched=True, features=features)
preprocessed_test_ds = test.map(preprocess_images, batched=True, features=features)

# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification2()

data_collator = default_data_collator

model.train()

search_space = {
    "learning_rate" : [2e-5],
    "batch_size" : [8],
    "weight_decay" : [0.01],
    "epochs" : [10],
}

hyperparamters_list = hyperparamters_from_dict(search_space)
training_args_list = parameters_to_trainingarguments(hyperparamters_list, path = path)

for configuration, args in training_args_list:
    model = ViTForImageClassification2()
    df = get_results_dataframe_if_exists(str(path) + '/results/grid_search_results_single_run.csv')
    identifier = shortuuid.uuid()

    trainer = Trainer(model,
        args,
        train_dataset = preprocessed_train_ds,
        eval_dataset = preprocessed_val_ds,
        data_collator = data_collator,
        compute_metrics = compute_metrics,
    )
    trainer.train()
    summary_results = merge_dicts({"id" : identifier}, configuration ,best_results_from_log(trainer.state.log_history))
    print(summary_results)
    all_results = merge_dicts(summary_results, {"all_results" : trainer.state.log_history , "model" : model})
    df = df.append(summary_results, ignore_index=True)
    to_pickle(file = all_results, name_location = str(path) + '/results/' + identifier + '.pkl')
    df.to_csv(path_or_buf = str(path) + '/results/grid_search_results_single_run.csv')
    print("Now running on Test set")
    trainer.evaluate(eval_dataset = preprocessed_test_ds)

# %%