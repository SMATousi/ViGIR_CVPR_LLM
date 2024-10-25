from snorkel.labeling import labeling_function
import json
import os
import numpy as np
from snorkel.labeling import LFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import LabelModel
from snorkel.analysis import metric_score
from snorkel.utils import probs_to_preds


POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1

@labeling_function()
def llava_7b(image_name):
    root_path = '/home/macula/SMATousi/CVPR/ViGIR_CVPR_LLM/prompting_framework/prompting_results/hateful/total_results/'
    llava_7b_results = 'llava:7b_results_hateful.json'
    path_to_llava_7b_results = os.path.join(root_path,llava_7b_results)
    with open(path_to_llava_7b_results, 'r') as file:
        data = json.load(file)

    return data[image_name] if data[image_name] is not None else -1

@labeling_function()
def llava_13b(image_name):
    root_path = '/home/macula/SMATousi/CVPR/ViGIR_CVPR_LLM/prompting_framework/prompting_results/hateful/total_results/'
    llava_7b_results = 'llava 13b-allsamples-results.json'
    path_to_llava_7b_results = os.path.join(root_path,llava_7b_results)
    with open(path_to_llava_7b_results, 'r') as file:
        data = json.load(file)

    return data[image_name] if data[image_name] is not None else -1

@labeling_function()
def bakllava(image_name):
    root_path = '/home/macula/SMATousi/CVPR/ViGIR_CVPR_LLM/prompting_framework/prompting_results/hateful/total_results/'
    llava_7b_results = 'bakllava-allsamples-results.json'
    path_to_llava_7b_results = os.path.join(root_path,llava_7b_results)
    with open(path_to_llava_7b_results, 'r') as file:
        data = json.load(file)

    return data[image_name] if data[image_name] is not None else -1

@labeling_function()
def llava_llama3(image_name):
    root_path = '/home/macula/SMATousi/CVPR/ViGIR_CVPR_LLM/prompting_framework/prompting_results/hateful/total_results/'
    llava_7b_results = 'llava-llama3-allsamples-results.json'
    path_to_llava_7b_results = os.path.join(root_path,llava_7b_results)
    with open(path_to_llava_7b_results, 'r') as file:
        data = json.load(file)

    return data[image_name] if data[image_name] is not None else -1

@labeling_function()
def llava_phi3(image_name):
    root_path = '/home/macula/SMATousi/CVPR/ViGIR_CVPR_LLM/prompting_framework/prompting_results/hateful/total_results/'
    llava_7b_results = 'llava-phi3-allsamples-results.json'
    path_to_llava_7b_results = os.path.join(root_path,llava_7b_results)
    with open(path_to_llava_7b_results, 'r') as file:
        data = json.load(file)

    return data[image_name] if data[image_name] is not None else -1


@labeling_function()
def moondream(image_name):
    root_path = '/home/macula/SMATousi/CVPR/ViGIR_CVPR_LLM/prompting_framework/prompting_results/hateful/total_results/'
    llava_7b_results = 'moondream-allsamples-results.json'
    path_to_llava_7b_results = os.path.join(root_path,llava_7b_results)
    with open(path_to_llava_7b_results, 'r') as file:
        data = json.load(file)

    return data[image_name] if data[image_name] is not None else -1