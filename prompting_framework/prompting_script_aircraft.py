import ollama 
import os
from tqdm import tqdm
import json
import signal
import argparse
import wandb
import pandas as pd
import sys
from  utilities import *
from PIL import Image
import matplotlib.pyplot as plt


def generate_context_embedding(class_names : str, model : str,  options : dict):
    """
        CVPR_W !!!
        - Function to create a context embedding for our given fine-grained class names!
        - We are interested in supplying this context to the VLM as it works on classifying
           images from a fine-grained dataset with numerous classes

        Inputs :
        -------
            class_names : str
                comma separated long string of class names.
                ex. class_names = "Honda accord, mazda rx9, mercedes benz c300"
            model       : str
                model being used in current experiment. 
                ex. if using 'llava-llama3' as the current vlm then we need to 
                use it as well for embedding extraction.
            options     : dict
                VLM options.
                ex. options= {  
                            "seed": 123,
                            "temperature": 0,
                            "num_ctx": 2048, # must be set, otherwise slightly random output
                        }
                
        Output :
        --------
            context_embedding : List
                vlm generated context embedding to aid in informed fine-grained classification.
                
    """
    prompt = "You are working on a difficult fine-grained image classification task with the following classes: " + class_names 
    context_response = ollama.generate(model=model, prompt=class_names, options=options)
    return context_response['context']


def main():


    parser = argparse.ArgumentParser(description="A script to run V-LLMs on Aircraft dataset")

    parser.add_argument("--modelname", type=str, required=True, help="The name of the V-LLM model")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt that you want to give to the V-LLM")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the image data dir")
    parser.add_argument("--json_file", type=str, required=True, help="json file to get the data names")
    parser.add_argument("--subset", type=str, required=True, help="train, test or validation set")
    parser.add_argument("--results_dir", type=str, required=True, help="Folder name to save results")
    parser.add_argument("--timeout", type=int, default=40, help="time out duration to skip one sample")
    parser.add_argument("--model_unloading", action="store_true", help="Enables unloading mode. Every 100 sampels it unloades the model from the GPU to avoid carshing.")


    args = parser.parse_args()

    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    run = wandb.init(
        # entity="jacketdembys",
        project=f"CVPR-2025-Aircraft",
        name="run_test_Aircraft_"+args.modelname+"-"+args.subset
    )

    class_names = "ATR, Airbus, Antonov, Beechcraft, Boeing, Bombardier Aerospace, British Aerospace, Canadair, Cessna, Cirrus Aircraft, Dassault Aviation, Dornier, Douglas Aircraft Company, Embraer, Eurofighter, Fairchild, Fokker, Gulfstream Aerospace, Ilyushin, Lockheed Corporation, Lockheed Martin, McDonnell Douglas, Panavia, Piper, Robin, Saab, Supermarine, Tupolev, Yakovlev, de Havilland"
    data ={}
    data_samples_path=args.data_samples
    print(data_samples_path)
    images_dir=os.path.join(args.data_path)
    print(images_dir)
    
    with open(data_samples_path, 'r') as file:
        raw_data_labels = json.load(file)
    
    raw_data=os.listdir(images_dir)
    for sample in raw_data_labels:
        #print(sample)
        data[os.path.join(images_dir,sample)]={"label":raw_data_labels[sample]["class_id"],"class":raw_data_labels[sample]["class_name"]}


    model_name = args.modelname
    results_dir=os.path.join(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    results_file_name=os.path.join(results_dir,f"{args.data}-{model_name}-{args.subset}.json")
    raw_image_info=os.path.join(results_dir,f"{args.data}-{model_name}-{args.subset}-raw_info.json")

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


    class_name_context = generate_context_embedding(class_names, model_name, options)


    for key,info in tqdm(data.items()):
        # print(type(key))
        count = count + 1
        image_path = key

        #disp_img(image_path)
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_duration)  

        try:
            if args.model_unloading and count % 99 == 0:
                response = ollama.generate(model=model_name, prompt=prompt, images=[image_path], options=options, keep_alive=0)
            else:
                response = ollama.generate(model=model_name, prompt=prompt, images=[image_path], options=options)
        
        except TimeoutException:
            print(f"Prompt for {image_path} took longer than {timeout_duration} seconds. Moving to the next one.")
            label = None
            
        finally:
            signal.alarm(0)  

        model_labels[image_path] = response['response']

    
    with open(results_file_name, 'w') as fp:
        json.dump(model_labels, fp, indent=4)
        
    with open(raw_image_info, 'w') as fp:
        json.dump(data, fp, indent=4)


    model_labels_df = pd.DataFrame(model_labels.items(), columns=["File Path", "Response"])
    model_labels_df["Image Name"] = model_labels_df["File Path"].apply(lambda x: x.split('/')[-1])
    model_labels_wandb = wandb.Table(data=model_labels_df)

    data_df = pd.DataFrame.from_dict(data, orient="index").reset_index()
    data_df.columns = ["File Path", "Label", "Class"]
    data_df_wandb = wandb.Table(data=data_df)

    wandb.log({"results": model_labels_wandb})
    wandb.log({"raw_images": data_df_wandb})

    # Optionally, if you want to save the JSON as an artifact
    artifact = wandb.Artifact("json_file", type="dataset")
    artifact.add_file(results_file_name)
    wandb.log_artifact(artifact)

    artifact = wandb.Artifact("json_file", type="dataset")
    artifact.add_file(raw_image_info)
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    main()

    
