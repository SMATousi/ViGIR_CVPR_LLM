import ollama 
import os
from tqdm import tqdm
import json


images_path = '/home1/goowfd/CVPR_2025/hateful_memes/img/'

list_of_image_names = os.listdir(images_path)

def check_yes_no(text):
    # Strip any leading/trailing whitespace and convert to lowercase
    text = text.strip().lower()

    # Check if the text starts with 'yes' or 'no'
    if text.startswith("yes"):
        return 1
    elif text.startswith("no"):
        return 0
    else:
        return None



model_name = 'llava:7b'
results_file_name = model_name + '_results_hateful.json'
ollama.pull(model_name)

root_path = '/home1/goowfd/CVPR_2025/hateful_memes/img/'

options= {  # new
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048, # must be set, otherwise slightly random output
        }
    
llava_7b_labels = {}

for image_name in tqdm(list_of_image_names):
    
    image_path = os.path.join(root_path, image_name)
    
    prompt = "Is this an offensive meme? Please answer with YES or NO. DO NOT mention the reason: "
#     prompt = "Is there ? Please answer with YES or NO. DO NOT mention the reason: "
#     prompt = "Is this somehow an offensive meme? Please answer with YES or NO: "
#     prompt = "describe this image: "
    
    response = ollama.generate(model='llava:7b', prompt=prompt, images=[image_path], options=options)
    

    label = check_yes_no(response['response'])
    
#     print(label)
    
    llava_7b_labels[image_name] = label
    
#     break

with open(results_file_name, 'w') as fp:
    json.dump(llava_7b_labels, fp)
