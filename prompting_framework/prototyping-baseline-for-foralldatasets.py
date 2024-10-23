#!/usr/bin/env python
# coding: utf-8

# In[2]:


import ollama 
import os
from tqdm import tqdm
import json
import signal
import argparse
import wandb

import sys


# In[3]:


from PIL import Image
import matplotlib.pyplot as plt


# In[4]:


# sys.argv = [
#     'notebook',  
#     '--modelname', 'llava:7b',
#     '--prompt', 'Is this an offensive meme? Please answer with YES or NO. DO NOT mention the reason:',
#     '--data', 'hateful_memes',
#     '--data_path','/home1/pupil/goowfd/CVPR_2025/hateful_memes/',
#     "--data_samples", 'train.jsonl',
#     '--results_dir', 'results/baselineExp',
#     '--timeout', '20',
#     '--model_unloading'
# ]


# In[5]:


parser = argparse.ArgumentParser(description="A script to run V-LLMs on different image classification datasets")


# In[6]:


parser.add_argument("--modelname", type=str, required=True, help="The name of the V-LLM model")
parser.add_argument("--prompt", type=str, required=True, help="The prompt that you want to give to the V-LLM")
parser.add_argument("--data", type=str, required=True, help="Dataset name")
parser.add_argument("--data_path", type=str, required=True, help="Path to the image data dir")
parser.add_argument("--data_samples", type=str, required=True, help="Name of the samples to run on")
parser.add_argument("--results_dir", type=str, required=True, help="Folder name to save results")
parser.add_argument("--timeout", type=int, default=40, help="time out duration to skip one sample")
parser.add_argument("--model_unloading", action="store_true", help="Enables unloading mode. Every 100 sampels it unloades the model from the GPU to avoid carshing.")

args = parser.parse_args()


# In[7]:


def check_yes_no(text):
    text = text.strip().lower()
    if text.startswith("yes"):
        return 1
    elif text.startswith("no"):
        return 0
    else:
        return None
    
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException
    
def disp_img(img_path): 
    img=Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')  
    plt.show()
    img.close()


# In[13]:


if args.data == 'hateful_memes':
    data =[]
    data_samples_path=os.path.join(args.data_path,args.data_samples)
    images_dir=os.path.join(args.data_path,'img')
    
    with open(data_samples_path, 'r') as file:
        for line in file: 
            record = json.loads(line)
            data.append({"path":os.path.join(images_dir,f'{record["img"][4:]}'),"label":record['label']})
            print(data[-1])

    
    


# In[9]:


model_name = args.modelname
results_dir=os.path.join(args.data_path, args.results_dir)
os.makedirs(results_dir, exist_ok=True)
results_file_name=os.path.join(results_dir,f"{model_name}.json")


# In[10]:


ollama.pull(model_name)

timeout_duration = args.timeout

options= {  # new
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048, # must be set, otherwise slightly random output
        }

model_labels = {}
prompt = args.prompt
count = 0


# In[14]:


for image_details in tqdm(data):

    image_path = image_details['path']


    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_duration)  

    try:
        response = ollama.generate(model=model_name, prompt=prompt, images=[image_path], options=options)
        label = check_yes_no(response['response'])
    except TimeoutException:
        print(f"Prompt for {image_name} took longer than {timeout_duration} seconds. Moving to the next one.")
        label = None
        
    finally:
        signal.alarm(0)  
    
    model_labels[image_path] = label   

    
    


# In[ ]:


with open(results_file_name, 'w') as fp:
    json.dump(model_labels, fp)


# In[ ]:




