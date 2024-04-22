import numpy as np
import os
import shutil
import json
import random
import torch
from numpyencoder import NumpyEncoder
import re


def set_random_seed(seed):
    """Sets random seed for training reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_dir_tree(base_dir, numfolds):
    dir_tree = ['models', 'config', 'std_log']
    last_run = 1
    for dir_ in dir_tree:
        if dir_ == dir_tree[0]:
            if not os.path.exists(os.path.join(base_dir)):
                os.makedirs(os.path.join(base_dir, str(last_run), dir_))
            else:
                last_run = np.max(list(map(int, os.listdir(base_dir))))
                last_run += 1
                if not os.path.exists(
                        os.path.join(base_dir, str(last_run - 1), 'classification_report_last.txt')):
                    last_run -= 1
                    shutil.rmtree(os.path.join(base_dir, str(last_run)))
                os.makedirs(os.path.join(base_dir, str(last_run), dir_))
        else:
            os.makedirs(os.path.join(base_dir, str(last_run), dir_))
    return last_run

def create_dir_tree2(base_dir, last_run):
    dir_tree = ['models', 'config']
    for dir_ in dir_tree:
        if dir_ == dir_tree[0]:
            if not os.path.exists(os.path.join(base_dir, str(last_run))):
                os.makedirs(os.path.join(base_dir, str(last_run), dir_))
            else:
                shutil.rmtree(os.path.join(base_dir, str(last_run)))
                os.makedirs(os.path.join(base_dir, str(last_run), dir_))
        else:
            os.makedirs(os.path.join(base_dir, str(last_run), dir_))
            
def save_json(filename, attributes, names):
    """
  Save training parameters and evaluation results to json file.
  :param filename: save filename
  :param attributes: attributes to save
  :param names: name of attributes to save in json file
  """
    with open(filename, "w", encoding="utf8") as outfile:
        d = {}
        for i in range(len(attributes)):
            name = names[i]
            attribute = attributes[i]
            d[name] = attribute
        json.dump(d, outfile, indent=4, cls=NumpyEncoder)
        
        
def is_substring(str1, str2):
    return str1.lower() in str2.lower()



def check_and_get_first_elements(list_of_lists):
    """
    check that all elements within each inner list are the same
    and also retrieve the first element of each inner list
    
    Parameters: list_of_lists (list of lists): A list containing inner lists to be checked.
    Returns: list: A list of the first elements from each uniform inner list.
    Raises: ValueError: If any inner list is empty or contains non-uniform elements.
    """
    first_elements = []

    for inner_list in list_of_lists:
        if not inner_list:
            raise ValueError("One of the inner lists is empty.")
        
        first_element = inner_list[0]
        if all(element == first_element for element in inner_list):
            first_elements.append(first_element)
        else:
            raise ValueError(f"Elements in the inner list {inner_list} differ.")

    return first_elements

def check_uniformity_and_get_first_elements(mainlist):
    try:
        mainlist = check_and_get_first_elements(mainlist)
        # print("First elements of each uniform inner list:", mainlist)
        return mainlist
    except ValueError as e:
        print(f"Error: {e}")
        


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]