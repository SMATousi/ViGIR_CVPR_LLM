import ollama
import os
from tqdm import tqdm
import json
import argparse
import wandb
import pandas as pd
import sys
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, CLIPModel 
from torch.nn.functional import cosine_similarity
import torch
import signal
from scipy.spatial.distance import cosine

# Example usage
"""
python run_traffic.py --base_dir /root/home/data/traffic/ --model_name "llava:7b" --prompt "Identify the traffic sign. Choose one
        from the list" --dataset_name "traffic" --data_path /root/home/data/traffic/images/ 
        --data_samples /root/home/data/traffic/annotations/traffic_signs_train.json --subset "train" 
        --timeout 20 --results_dir /root/home/data/traffic/results/ 
"""



def get_class_embeddings(prompts, tokenizer, text_encoder):
    text_inputs = tokenizer(prompts, padding="max_length", return_tensors="pt").to(device)
    outputs = text_encoder(**text_inputs)
    text_embedding = outputs.pooler_output
    return text_embedding
    
def get_query_embedding(query_prompt, tokenizer, text_encoder):
    
    query_input = tokenizer(query_prompt, padding="max_length", return_tensors="pt").to(device)
    query_output = text_encoder(**query_input)
    query_embedding = query_output.pooler_output
    return query_embedding

def compute_scores(class_embeddings, query_embedding, prompts):
     # Compute cosine similarity scores
    similarity_scores = cosine_similarity(query_embedding, class_embeddings, dim=1)  # Shape: [37]
    
    # Find the highest matching score and corresponding item
    max_score_index = torch.argmax(similarity_scores).item()
    max_score = similarity_scores[max_score_index].item()
    best_match = prompts[max_score_index]
    
    # Print the result
   # print(f"Best match: {best_match} with a similarity score of {max_score:.4f}")
    return best_match

def generate_context_embedding(class_names, model_name, options):
    prompt = "You are working on a difficult fine-grained image classification task, here are the only classes you can choose from"+class_names
    context_response = ollama.generate(model=model_name, prompt=prompt, options=options)
    return context_response['context']

def compute_class_embeddings(class_names_list) :
    class_embeddings = {}
    for class_name in class_names_list :
        # print(class_name)
        response = ollama.embed(model="mxbai-embed-large", input=class_name)
        class_embeddings[class_name] = response["embeddings"]
        return class_embeddings


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException


parser = argparse.ArgumentParser(description="Run numerous experiments with varying VLM models on the traffic sign dataset")

parser.add_argument("--base_dir", type=str, required=True, help="Base dataset directory")
parser.add_argument("--model_name", type=str, required=True, help=" VLM model name")
parser.add_argument("--prompt", type=str, required=True, help="VLM prompt")
parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
parser.add_argument("--data_path", type=str, required=True, help="Path to the image data dir")
parser.add_argument("--data_samples", type=str, required=True, help="JSON file with data samples to run")
parser.add_argument("--subset", type=str, required=True, help="train, test or validation set")
parser.add_argument("--results_dir", type=str, required=True, help="Folder name to save results")
parser.add_argument("--timeout", type=int, default=40, help="time out duration to skip one sample")
parser.add_argument("--model_unloading", action="store_true", help="Enables unloading mode. Every 100 sampels it unloades the model from the GPU to avoid crashing.")


args = parser.parse_args()
#dataset_name = "traffic"
#model_unloading= True
if args.dataset_name == "traffic":
    class_names = "Speed limit (20km/h), Speed limit (30km/h), Speed limit (50km/h), Speed limit (60km/h), Speed limit (70km/h), Speed limit (80km/h), End of speed limit (80km/h), Speed limit (100km/h), Speed limit (120km/h), No passing, No passing for vehicles over 3.5 metric tons, Right-of-way at the next intersection, Priority road, Yield, Stop, No vehicles, Vehicles over 3.5 metric tons prohibited, No entry, General caution, Dangerous curve to the left, Dangerous curve to the right, Double curve, Bumpy road, Slippery road, Road narrows on the right, Road work, Traffic signals, Pedestrians, Children crossing, Bicycles crossing, Beware of ice/snow, Wild animals crossing, End of all speed and passing limits, Turn right ahead, Turn left ahead, Ahead only, Go straight or right, Go straight or left, Keep right, Keep left, Roundabout mandatory, End of no passing, End of no passing by vehicles over 3.5 metric tons"

    data = {}
    base_dir = args.base_dir #"/root/home/data/traffic/"
    data_samples_file_path = args.data_samples #"/root/home/data/traffic/annotations/traffic_signs_train.json"
    data_path = args.data_path#"/root/home/data/traffic/images/" #args.data_path
    images_dir = os.path.join(data_path) # /root/home/data/traffic/images/ -- need train/0/

    with open(data_samples_file_path, 'r') as file:
        raw_data = json.load(file)

    for image_path, details in raw_data.items():
        class_id = details["class_id"]
        class_name = details["class_name"]
        image_file_path = image_path.lower()
        
        data[os.path.join(images_dir, image_file_path)] = {"label" : class_id, "class" : class_name}

    model_name  = args.model_name
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    dataset_name = args.dataset_name#"traffic"
    subset = args.subset
    results_file_name=os.path.join(results_dir,f"{dataset_name}-{model_name}-{subset}.json")
    raw_image_info=os.path.join(results_dir,f"{dataset_name}-{model_name}-{subset}-raw_info.json")
    print("Pulling Ollama Model...")
    print(model_name)
    ollama.pull(model_name)
    print("Done Pulling..")
    timeout_duration = args.timeout

    options= {  # new
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048, # must be set, otherwise slightly random output
        }

    # model_id_clip  = "openai/clip-vit-large-patch14"
    # device="cuda" if torch.cuda.is_available() else "cpu"
    # print("Setting up CLIP..")

    # tokenizer = CLIPTokenizer.from_pretrained(model_id_clip)
    # text_encoder = CLIPTextModel.from_pretrained(model_id_clip).to(device)
    # clip_model = CLIPModel.from_pretrained(model_id_clip).to(device)

    class_names_list = [name.strip() for name in class_names.split(',')]
    class_dict = {class_name : i for i, class_name in enumerate(class_names_list)}
    ollama.pull("mxbai-embed-large") # model for embedding class names text
    class_embeddings = compute_class_embeddings(class_names_list)
    #traffic_embeedings = get_class_embeddings(class_names_list, tokenizer, text_encoder)
    context_embedding = generate_context_embedding(class_names, model_name, options)
    # print("Done setting up clip...")
    model_labels = {}
    prompt = args.prompt
    count = 0
    
    print("Begin prompting...")

    for key,info in tqdm(data.items()):
        # print(type(key))
        count = count + 1
        image_path = key

    #disp_img(image_path)

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_duration)

        try:
            if args.model_unloading and count % 99 == 0:
                response = ollama.generate(model=model_name, prompt=prompt, images=[image_path], options=options,context=context_embedding, keep_alive=0)
            else:
                response = ollama.generate(model=model_name, prompt=prompt, images=[image_path], options=options, context=context_embedding)

        except TimeoutException:
            print(f"Prompt for {image_path} took longer than {timeout_duration} seconds. Moving to the next one.")
            label = None

        finally:
            signal.alarm(0)

        model_response = response['response']
        query_response = ollama.embed(model="mxbai-embed-large", input=model_response)
        query_embedding = query_response["embeddings"]
        # Initialize variables for the best match
        best_match = None
        best_similarity = -1  # Cosine similarity ranges from -1 to 1, so start with a very low value
        
        # Find the best matching embedding
        for class_name, class_embedding in class_embeddings.items():
            similarity = 1 - cosine(query_embedding[0], class_embedding[0])  # Cosine similarity is 1 - cosine distance
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = class_name
        #= get_query_embedding(model_response, tokenizer, text_encoder)
        matched_label = best_match #compute_scores(traffic_embeedings, response_embedding, class_names_list)
        class_label = class_dict[matched_label]
       # print(f"{image_path} | {matched_label} | {model_response}")
        model_labels[image_path] = {
            "label": class_label, # integer index representing class
            "class": matched_label, # string indicating class name
            "model_response": model_response # string coming from the model
        }

    with open(results_file_name, 'w') as fp:
        json.dump(model_labels, fp, indent=4)

    with open(raw_image_info, 'w') as fp:
        json.dump(data, fp, indent=4)
