import ollama 
import os
from tqdm import tqdm
import json
import signal



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
    
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException



root_path = '/home1/pupil/goowfd/CVPR_2025/hateful_memes/img'

list_of_image_names = os.listdir(root_path)

model_name = 'llava:7b'
results_file_name = model_name + '_results_hateful.json'
ollama.pull(model_name)

timeout_duration = 20
print(f"Handling the timeout exceptions with timeout duration of {timeout_duration} seconds")
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
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_duration)  # Set the timeout

    try:
        response = ollama.generate(model='llava:7b', prompt=prompt, images=[image_path], options=options)

        label = check_yes_no(response['response'])

    except TimeoutException:
        print(f"Prompt for {image_name} took longer than {timeout_duration} seconds. Moving to the next one.")
        label = None

    finally:
        signal.alarm(0)  # Disable the alarm

    
    
#     print(label)
    
    llava_7b_labels[image_name] = label
    
#     break

with open(results_file_name, 'w') as fp:
    json.dump(llava_7b_labels, fp)
