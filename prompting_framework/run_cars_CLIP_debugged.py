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
import torch.nn.functional as F
from scipy.spatial.distance import cosine



# def get_class_embeddings(prompts, tokenizer, text_encoder):
#     text_inputs = tokenizer(prompts, padding="max_length", return_tensors="pt").to(device)
#     outputs = text_encoder(**text_inputs)
#     text_embedding = outputs.pooler_output
#     return text_embedding
    
# def get_query_embedding(query_prompt, tokenizer, text_encoder):
    
#     query_input = tokenizer(query_prompt, padding="max_length", return_tensors="pt").to(device)
#     query_output = text_encoder(**query_input)
#     query_embedding = query_output.pooler_output
#     return query_embedding

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

def compute_scores_clip(class_embeddings, query_embedding, prompts):
     # Compute cosine similarity scores
    similarity_scores = cosine_similarity(query_embedding, class_embeddings, dim=1)  # Shape: [37]
    
    # Find the highest matching score and corresponding item
    max_score_index = torch.argmax(similarity_scores).item()
    max_score = similarity_scores[max_score_index].item()
    best_match = prompts[max_score_index]
    
    # Print the result
   # print(f"Best match: {best_match} with a similarity score of {max_score:.4f}")
    return best_match

    
def compute_best_match(query_text, class_embeddings, class_dict, model_name):
    query_response = ollama.embed(model=model_name, input=query_text)
    query_embedding = query_response["embeddings"]
    query_embedding_tensor = torch.tensor(query_embedding[0])
    
    list_name_emb = list(class_embeddings.keys())
    current_best_score = -1.0  # Start with a low value for cosine similarity
    current_best_match = ""
    
    for class_name in list_name_emb:
        #print(f"Comparing with class: {class_name}")
        class_embeddings_tensor = torch.tensor(class_embeddings[class_name][0])
        # Compute the cosine similarity
        similarity_score = F.cosine_similarity(query_embedding_tensor.unsqueeze(0), class_embeddings_tensor.unsqueeze(0), dim=1)
        
        if similarity_score > current_best_score:
            current_best_score = similarity_score.item()  # Ensure it's a Python float for printing
            current_best_match = class_name
            #print(f"Current best match is: {current_best_match} with score: {current_best_score}")
    matched_label = class_dict[current_best_match] # integer representing class 
    matched_class_name  = current_best_match
    return matched_class_name, matched_label 

def compute_scores(class_embeddings, query_embedding, prompts, temperature=0.8):
    scores = []
    # Compute cosine similarity scores
    for class_name in class_embeddings:
        similarity_scores = cosine_similarity(torch.tensor(query_embedding), torch.tensor(class_embeddings[class_name]), dim=1)  # Shape: [37]
        similarity_scores = similarity_scores / temperature
        scores.append(similarity_scores.item())

    probabilities = F.softmax(torch.tensor(scores), dim=0)
    # Find the highest matching score and corresponding item

    max_prob_index = torch.argmax(probabilities).item()
    max_prob = probabilities[max_prob_index]
    best_match = prompts[max_prob_index]
    
    # Print the result
   # print(f"Best match: {best_match} with a similarity score of {max_score:.4f}")
    return best_match, probabilities, max_prob

def generate_context_embedding(class_names, model_name, options):
    prompt = "You are working on a difficult fine-grained image classification task, here are the only classes you can choose from"+class_names
    context_response = ollama.generate(model=model_name, prompt=prompt, options=options)
    return context_response['context']

def compute_class_embeddings(class_names_list, model_name) :
    class_embeddings = {}
    print("Computing the class embeddings --")
    for class_name in tqdm(class_names_list) :
        # print(class_name)
        response = ollama.embed(model=model_name, input=class_name)
        class_embeddings[class_name] = response["embeddings"]
    
    return class_embeddings


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException


parser = argparse.ArgumentParser(description="Run numerous experiments with varying VLM models on the Cars sign dataset")

parser.add_argument("--base_dir", type=str, required=False, help="Base dataset directory")
parser.add_argument("--model_name", type=str, required=True, help=" VLM model name")
parser.add_argument("--prompt", type=str, required=False, help="VLM prompt")
parser.add_argument("--dataset_name", type=str, required=False, help="Dataset name")
parser.add_argument("--data_path", type=str, required=False, help="Path to the image data dir")
parser.add_argument("--data_samples", type=str, required=False, help="JSON file with data samples to run")
parser.add_argument("--subset", type=str, required=True, help="train, test or validation set")
parser.add_argument("--results_dir", type=str, required=False, help="Folder name to save results")
parser.add_argument("--timeout", type=int, default=100, help="time out duration to skip one sample")
parser.add_argument("--model_unloading", action="store_true", help="Enables unloading mode. Every 100 sampels it unloades the model from the GPU to avoid crashing.")


args = parser.parse_args()
dataset_name = "cars"

#model_unloading= True

run = wandb.init(
    # entity="jacketdembys",
    project=f"CVPR-2025-Cars-CLIP",
    name=f"run_test_{dataset_name}_"+args.model_name+"-"+args.subset
)
table = wandb.Table(columns=["image_path", "pred_class", "pred_class_ind", "pred_response"])

if dataset_name == "cars":
    class_names = 'AM General Hummer SUV 2000, Acura Integra Type R 2001, Acura RL Sedan 2012, Acura TL Sedan 2012, Acura TL Type-S 2008, Acura TSX Sedan 2012, Acura ZDX Hatchback 2012, Aston Martin V8 Vantage Convertible 2012, Aston Martin V8 Vantage Coupe 2012, Aston Martin Virage Convertible 2012, Aston Martin Virage Coupe 2012, Audi 100 Sedan 1994, Audi 100 Wagon 1994, Audi A5 Coupe 2012, Audi R8 Coupe 2012, Audi RS 4 Convertible 2008, Audi S4 Sedan 2007, Audi S4 Sedan 2012, Audi S5 Convertible 2012, Audi S5 Coupe 2012, Audi S6 Sedan 2011, Audi TT Hatchback 2011, Audi TT RS Coupe 2012, Audi TTS Coupe 2012, Audi V8 Sedan 1994, BMW 1 Series Convertible 2012, BMW 1 Series Coupe 2012, BMW 3 Series Sedan 2012, BMW 3 Series Wagon 2012, BMW 6 Series Convertible 2007, BMW ActiveHybrid 5 Sedan 2012, BMW M3 Coupe 2012, BMW M5 Sedan 2010, BMW M6 Convertible 2010, BMW X3 SUV 2012, BMW X5 SUV 2007, BMW X6 SUV 2012, BMW Z4 Convertible 2012, Bentley Arnage Sedan 2009, Bentley Continental Flying Spur Sedan 2007, Bentley Continental GT Coupe 2007, Bentley Continental GT Coupe 2012, Bentley Continental Supersports Conv. Convertible 2012, Bentley Mulsanne Sedan 2011, Bugatti Veyron 16.4 Convertible 2009, Bugatti Veyron 16.4 Coupe 2009, Buick Enclave SUV 2012, Buick Rainier SUV 2007, Buick Regal GS 2012, Buick Verano Sedan 2012, Cadillac CTS-V Sedan 2012, Cadillac Escalade EXT Crew Cab 2007, Cadillac SRX SUV 2012, Chevrolet Avalanche Crew Cab 2012, Chevrolet Camaro Convertible 2012, Chevrolet Cobalt SS 2010, Chevrolet Corvette Convertible 2012, Chevrolet Corvette Ron Fellows Edition Z06 2007, Chevrolet Corvette ZR1 2012, Chevrolet Express Cargo Van 2007, Chevrolet Express Van 2007, Chevrolet HHR SS 2010, Chevrolet Impala Sedan 2007, Chevrolet Malibu Hybrid Sedan 2010, Chevrolet Malibu Sedan 2007, Chevrolet Monte Carlo Coupe 2007, Chevrolet Silverado 1500 Classic Extended Cab 2007, Chevrolet Silverado 1500 Extended Cab 2012, Chevrolet Silverado 1500 Hybrid Crew Cab 2012, Chevrolet Silverado 1500 Regular Cab 2012, Chevrolet Silverado 2500HD Regular Cab 2012, Chevrolet Sonic Sedan 2012, Chevrolet Tahoe Hybrid SUV 2012, Chevrolet TrailBlazer SS 2009, Chevrolet Traverse SUV 2012, Chrysler 300 SRT-8 2010, Chrysler Aspen SUV 2009, Chrysler Crossfire Convertible 2008, Chrysler PT Cruiser Convertible 2008, Chrysler Sebring Convertible 2010, Chrysler Town and Country Minivan 2012, Daewoo Nubira Wagon 2002, Dodge Caliber Wagon 2007, Dodge Caliber Wagon 2012, Dodge Caravan Minivan 1997, Dodge Challenger SRT8 2011, Dodge Charger SRT-8 2009, Dodge Charger Sedan 2012, Dodge Dakota Club Cab 2007, Dodge Dakota Crew Cab 2010, Dodge Durango SUV 2007, Dodge Durango SUV 2012, Dodge Journey SUV 2012, Dodge Magnum Wagon 2008, Dodge Ram Pickup 3500 Crew Cab 2010, Dodge Ram Pickup 3500 Quad Cab 2009, Dodge Sprinter Cargo Van 2009, Eagle Talon Hatchback 1998, FIAT 500 Abarth 2012, FIAT 500 Convertible 2012, Ferrari 458 Italia Convertible 2012, Ferrari 458 Italia Coupe 2012, Ferrari California Convertible 2012, Ferrari FF Coupe 2012, Fisker Karma Sedan 2012, Ford E-Series Wagon Van 2012, Ford Edge SUV 2012, Ford Expedition EL SUV 2009, Ford F-150 Regular Cab 2007, Ford F-150 Regular Cab 2012, Ford F-450 Super Duty Crew Cab 2012, Ford Fiesta Sedan 2012, Ford Focus Sedan 2007, Ford Freestar Minivan 2007, Ford GT Coupe 2006, Ford Mustang Convertible 2007, Ford Ranger SuperCab 2011, GMC Acadia SUV 2012, GMC Canyon Extended Cab 2012, GMC Savana Van 2012, GMC Terrain SUV 2012, GMC Yukon Hybrid SUV 2012, Geo Metro Convertible 1993, HUMMER H2 SUT Crew Cab 2009, HUMMER H3T Crew Cab 2010, Honda Accord Coupe 2012, Honda Accord Sedan 2012, Honda Odyssey Minivan 2007, Honda Odyssey Minivan 2012, Hyundai Accent Sedan 2012, Hyundai Azera Sedan 2012, Hyundai Elantra Sedan 2007, Hyundai Elantra Touring Hatchback 2012, Hyundai Genesis Sedan 2012, Hyundai Santa Fe SUV 2012, Hyundai Sonata Hybrid Sedan 2012, Hyundai Sonata Sedan 2012, Hyundai Tucson SUV 2012, Hyundai Veloster Hatchback 2012, Hyundai Veracruz SUV 2012, Infiniti G Coupe IPL 2012, Infiniti QX56 SUV 2011, Isuzu Ascender SUV 2008, Jaguar XK XKR 2012, Jeep Compass SUV 2012, Jeep Grand Cherokee SUV 2012, Jeep Liberty SUV 2012, Jeep Patriot SUV 2012, Jeep Wrangler SUV 2012, Lamborghini Aventador Coupe 2012, Lamborghini Diablo Coupe 2001, Lamborghini Gallardo LP 570-4 Superleggera 2012, Lamborghini Reventon Coupe 2008, Land Rover LR2 SUV 2012, Land Rover Range Rover SUV 2012, Lincoln Town Car Sedan 2011, MINI Cooper Roadster Convertible 2012, Maybach Landaulet Convertible 2012, Mazda Tribute SUV 2011, McLaren MP4-12C Coupe 2012, Mercedes-Benz 300-Class Convertible 1993, Mercedes-Benz C-Class Sedan 2012, Mercedes-Benz E-Class Sedan 2012, Mercedes-Benz S-Class Sedan 2012, Mercedes-Benz SL-Class Coupe 2009, Mercedes-Benz Sprinter Van 2012, Mitsubishi Lancer Sedan 2012, Nissan 240SX Coupe 1998, Nissan Juke Hatchback 2012, Nissan Leaf Hatchback 2012, Nissan NV Passenger Van 2012, Plymouth Neon Coupe 1999, Porsche Panamera Sedan 2012, Ram C/V Cargo Van Minivan 2012, Rolls-Royce Ghost Sedan 2012, Rolls-Royce Phantom Drophead Coupe Convertible 2012, Rolls-Royce Phantom Sedan 2012, Scion xD Hatchback 2012, Spyker C8 Convertible 2009, Spyker C8 Coupe 2009, Suzuki Aerio Sedan 2007, Suzuki Kizashi Sedan 2012, Suzuki SX4 Hatchback 2012, Suzuki SX4 Sedan 2012, Tesla Model S Sedan 2012, Toyota 4Runner SUV 2012, Toyota Camry Sedan 2012, Toyota Corolla Sedan 2012, Toyota Sequoia SUV 2012, Volkswagen Beetle Hatchback 2012, Volkswagen Golf Hatchback 1991, Volkswagen Golf Hatchback 2012, Volvo 240 Sedan 1993, Volvo C30 Hatchback 2012, Volvo XC90 SUV 2007, smart fortwo Convertible 2012'    
    data = {}
    base_dir = f"/root/home/data/{dataset_name}/" #args.base_dir #
    data_samples_file_path = f"/root/home/data/{dataset_name}/annotations/stanford_{dataset_name}_{args.subset}.json" # args.data_samples #
    data_path = f"/root/home/data/{dataset_name}/images/{args.subset}" #args.data_path
    images_dir =  f"/root/home/data/{dataset_name}/images/{args.subset}" #-- need train/0/ os.path.join(data_path) #

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

    model_id_clip  = "openai/clip-vit-large-patch14"
    device="cuda" if torch.cuda.is_available() else "cpu"
    print("Setting up CLIP..")

    tokenizer = CLIPTokenizer.from_pretrained(model_id_clip)
    text_encoder = CLIPTextModel.from_pretrained(model_id_clip).to(device)
    clip_model = CLIPModel.from_pretrained(model_id_clip).to(device)

    class_names_list = [name.strip() for name in class_names.split(',')]
    class_dict = {class_name : i for i, class_name in enumerate(class_names_list)}
    reverse_class_dict = {v: k for k, v in class_dict.items()}
    # ollama.pull("mxbai-embed-large") # model for embedding class names text
    #class_embeddings = compute_class_embeddings(class_names_list, model_name)
    class_embeddings = get_class_embeddings(class_names_list, tokenizer, text_encoder)
    context_embedding = generate_context_embedding(class_names, model_name, options)
    # print("Done setting up clip...")
    model_labels = {}
    prompt = args.prompt
    count = 0
    
    print("Begin prompting...")
    text_length = 500
    for key,info in tqdm(data.items()):
        # print(type(key))
        count = count + 1
        image_path = key
        print(f"Current count {count}")
        #if count  > 10 : 
           # break
    #disp_img(image_path)

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_duration)

        try:
        
            response = ollama.generate(model=model_name, prompt=prompt, images=[image_path], options=options, context=context_embedding)
            # response = ollama.generate(model=model_name, prompt=prompt, options=options, context=context_embedding)

        

            model_response = response['response']
            if len(model_response) > text_length : 
                query_prompt = model_response[:text_length]
            else : 
                query_prompt = model_response
            query_embedding = get_query_embedding(query_prompt, tokenizer, text_encoder)
            class_name = compute_scores_clip(class_embeddings, query_embedding, class_names_list)
            class_number = class_dict[class_name]
            
            #query_response = ollama.embed(model=model_name, input=model_response)
            #query_embedding = query_response["embeddings"]
            # print(query_embedding)

            #best_match_class, best_match_label = compute_best_match(model_response, class_embeddings, class_dict,model_name)

            #best_match, probs, max_prob = compute_scores(class_embeddings, query_embedding[0], class_names_list, temperature=0.2)
            #class_label =  best_match_label #class_dict[best_match]
        
            # Initialize variables for the best match
            # best_match = None
            # best_similarity = -1  # Cosine similarity ranges from -1 to 1, so start with a very low value
            
            # # Find the best matching embedding
            # for class_name, class_embedding in class_embeddings.items():
            #     similarity = 1 - cosine(query_embedding[0], class_embedding[0])  # Cosine similarity is 1 - cosine distance
            #     if similarity > best_similarity:
            #         best_similarity = similarity
            #         best_match = class_name
            # #= get_query_embedding(model_response, tokenizer, text_encoder)
            # matched_label = best_match #compute_scores(traffic_embeedings, response_embedding, class_names_list)
            
        # print(f"{image_path} | {matched_label} | {model_response}")
            model_labels[image_path] = {
                "label": class_number, # integer index representing class
                "class": class_name, # string indicating class name
                "model_response": model_response, # string coming from the model
                "query_prompt": query_prompt # actual prompt used to generate embedding
            }
           
            table.add_data(image_path, class_name, class_number, model_response)

            wandb.log({"Results": table})

            # print(model_labels)
            # wandb.log(model_labels)

        except (TimeoutException, ValueError) as e:
            print(e)
            print(f"Prompt for {image_path} took longer than {timeout_duration} seconds. Moving to the next one.")
            model_labels[image_path] = {
                "label": None, # integer index representing class
                "class": None, # string indicating class name
                "model_response": None # string coming from the model
            }
            pass

        finally:
            signal.alarm(0)
            # model_labels[image_path] = {
            #     "label": class_label, # integer index representing class
            #     "class": best_match, # string indicating class name
            #     "model_response": model_response # string coming from the model
            # }

            

    with open(results_file_name, 'w') as fp:
        json.dump(model_labels, fp, indent=4)

    with open(raw_image_info, 'w') as fp:
        json.dump(data, fp, indent=4)



    # Optionally, if you want to save the JSON as an artifact
    artifact = wandb.Artifact("json_file", type="dataset")
    artifact.add_file(results_file_name)
    wandb.log_artifact(artifact)

    artifact = wandb.Artifact("json_file", type="dataset")
    artifact.add_file(raw_image_info)
    wandb.log_artifact(artifact)