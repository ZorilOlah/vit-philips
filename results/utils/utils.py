import pickle
from typing import List, Set, Dict, Tuple, Optional
import pandas as pd
import os
from pathlib import Path

def to_pickle(file, name_location):
    with open(name_location, "ab") as f:
        pickle.dump(file, f)
    print(f"File Pickled at {name_location}")


def load_pickle(name_location):
    with open(name_location, "rb") as f:
        db = pickle.load(f)
    print(f"Loaded Pickle {name_location}")
    return db

def merge_dicts(*dicts : dict):
    new_dict = {}
    for dict in dicts:
        new_dict.update(dict)
    return new_dict

def get_results_dataframe_if_exists(path : str):
    if Path(path).is_file():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()
    return df

def best_results_from_log(log_list : List) -> float:
    best_acc = 0
    best_epoch = 0
    for epoch in log_list:
        if 'eval_accuracy' in epoch:
            acc = epoch['eval_accuracy']
            epoch = epoch['epoch']
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
    return {'best_epoch' : best_epoch, 'best_acc' : best_acc}

