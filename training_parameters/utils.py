#%%
import itertools
from datasets import load_metric
import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from pathlib import Path
from transformers import ViTModel, ViTConfig, ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments


def hyperparamters_from_dict(hyperparamter_dict : dict):
    hyper_parameters = []
    for values in itertools.product(*hyperparamter_dict.values()):
        hyper_parameters.append(dict(zip(hyperparamter_dict.keys(), values)))
    return hyper_parameters

def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def make_training_object(learning_rate : float, batch_size : int, weight_decay : float, epochs : int, path : Path):
        training_args = TrainingArguments(
            evaluation_strategy = "epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            load_best_model_at_end=False,
            metric_for_best_model='accuracy',
            logging_dir= str(path) + '/logs/huggingface_logs',
            logging_strategy="epoch",
            remove_unused_columns=True,
            output_dir = str(path) + '/results/huggingface_results'
        )
        return(training_args)

def parameters_to_trainingarguments(hyperparameters_dict_list : List, path : Path):
    training_args_list = []
    for configuration in hyperparameters_dict_list:
        training_args_list.append((configuration, make_training_object(
            learning_rate = configuration["learning_rate"],
            batch_size = configuration["batch_size"],
            weight_decay = configuration["weight_decay"],
            epochs = configuration["epochs"],
            path = path
            )))
    return(training_args_list)
# %%
