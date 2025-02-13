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


    parser = argparse.ArgumentParser(description="A script to run V-LLMs on Cars dataset")

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
        project=f"CVPR-2025-Cars",
        name="run_test_cars_"+args.modelname+"-"+args.subset
    )

    class_names = 'AM General Hummer SUV 2000, Acura Integra Type R 2001, Acura RL Sedan 2012, Acura TL Sedan 2012, Acura TL Type-S 2008, Acura TSX Sedan 2012, Acura ZDX Hatchback 2012, Aston Martin V8 Vantage Convertible 2012, Aston Martin V8 Vantage Coupe 2012, Aston Martin Virage Convertible 2012, Aston Martin Virage Coupe 2012, Audi 100 Sedan 1994, Audi 100 Wagon 1994, Audi A5 Coupe 2012, Audi R8 Coupe 2012, Audi RS 4 Convertible 2008, Audi S4 Sedan 2007, Audi S4 Sedan 2012, Audi S5 Convertible 2012, Audi S5 Coupe 2012, Audi S6 Sedan 2011, Audi TT Hatchback 2011, Audi TT RS Coupe 2012, Audi TTS Coupe 2012, Audi V8 Sedan 1994, BMW 1 Series Convertible 2012, BMW 1 Series Coupe 2012, BMW 3 Series Sedan 2012, BMW 3 Series Wagon 2012, BMW 6 Series Convertible 2007, BMW ActiveHybrid 5 Sedan 2012, BMW M3 Coupe 2012, BMW M5 Sedan 2010, BMW M6 Convertible 2010, BMW X3 SUV 2012, BMW X5 SUV 2007, BMW X6 SUV 2012, BMW Z4 Convertible 2012, Bentley Arnage Sedan 2009, Bentley Continental Flying Spur Sedan 2007, Bentley Continental GT Coupe 2007, Bentley Continental GT Coupe 2012, Bentley Continental Supersports Conv. Convertible 2012, Bentley Mulsanne Sedan 2011, Bugatti Veyron 16.4 Convertible 2009, Bugatti Veyron 16.4 Coupe 2009, Buick Enclave SUV 2012, Buick Rainier SUV 2007, Buick Regal GS 2012, Buick Verano Sedan 2012, Cadillac CTS-V Sedan 2012, Cadillac Escalade EXT Crew Cab 2007, Cadillac SRX SUV 2012, Chevrolet Avalanche Crew Cab 2012, Chevrolet Camaro Convertible 2012, Chevrolet Cobalt SS 2010, Chevrolet Corvette Convertible 2012, Chevrolet Corvette Ron Fellows Edition Z06 2007, Chevrolet Corvette ZR1 2012, Chevrolet Express Cargo Van 2007, Chevrolet Express Van 2007, Chevrolet HHR SS 2010, Chevrolet Impala Sedan 2007, Chevrolet Malibu Hybrid Sedan 2010, Chevrolet Malibu Sedan 2007, Chevrolet Monte Carlo Coupe 2007, Chevrolet Silverado 1500 Classic Extended Cab 2007, Chevrolet Silverado 1500 Extended Cab 2012, Chevrolet Silverado 1500 Hybrid Crew Cab 2012, Chevrolet Silverado 1500 Regular Cab 2012, Chevrolet Silverado 2500HD Regular Cab 2012, Chevrolet Sonic Sedan 2012, Chevrolet Tahoe Hybrid SUV 2012, Chevrolet TrailBlazer SS 2009, Chevrolet Traverse SUV 2012, Chrysler 300 SRT-8 2010, Chrysler Aspen SUV 2009, Chrysler Crossfire Convertible 2008, Chrysler PT Cruiser Convertible 2008, Chrysler Sebring Convertible 2010, Chrysler Town and Country Minivan 2012, Daewoo Nubira Wagon 2002, Dodge Caliber Wagon 2007, Dodge Caliber Wagon 2012, Dodge Caravan Minivan 1997, Dodge Challenger SRT8 2011, Dodge Charger SRT-8 2009, Dodge Charger Sedan 2012, Dodge Dakota Club Cab 2007, Dodge Dakota Crew Cab 2010, Dodge Durango SUV 2007, Dodge Durango SUV 2012, Dodge Journey SUV 2012, Dodge Magnum Wagon 2008, Dodge Ram Pickup 3500 Crew Cab 2010, Dodge Ram Pickup 3500 Quad Cab 2009, Dodge Sprinter Cargo Van 2009, Eagle Talon Hatchback 1998, FIAT 500 Abarth 2012, FIAT 500 Convertible 2012, Ferrari 458 Italia Convertible 2012, Ferrari 458 Italia Coupe 2012, Ferrari California Convertible 2012, Ferrari FF Coupe 2012, Fisker Karma Sedan 2012, Ford E-Series Wagon Van 2012, Ford Edge SUV 2012, Ford Expedition EL SUV 2009, Ford F-150 Regular Cab 2007, Ford F-150 Regular Cab 2012, Ford F-450 Super Duty Crew Cab 2012, Ford Fiesta Sedan 2012, Ford Focus Sedan 2007, Ford Freestar Minivan 2007, Ford GT Coupe 2006, Ford Mustang Convertible 2007, Ford Ranger SuperCab 2011, GMC Acadia SUV 2012, GMC Canyon Extended Cab 2012, GMC Savana Van 2012, GMC Terrain SUV 2012, GMC Yukon Hybrid SUV 2012, Geo Metro Convertible 1993, HUMMER H2 SUT Crew Cab 2009, HUMMER H3T Crew Cab 2010, Honda Accord Coupe 2012, Honda Accord Sedan 2012, Honda Odyssey Minivan 2007, Honda Odyssey Minivan 2012, Hyundai Accent Sedan 2012, Hyundai Azera Sedan 2012, Hyundai Elantra Sedan 2007, Hyundai Elantra Touring Hatchback 2012, Hyundai Genesis Sedan 2012, Hyundai Santa Fe SUV 2012, Hyundai Sonata Hybrid Sedan 2012, Hyundai Sonata Sedan 2012, Hyundai Tucson SUV 2012, Hyundai Veloster Hatchback 2012, Hyundai Veracruz SUV 2012, Infiniti G Coupe IPL 2012, Infiniti QX56 SUV 2011, Isuzu Ascender SUV 2008, Jaguar XK XKR 2012, Jeep Compass SUV 2012, Jeep Grand Cherokee SUV 2012, Jeep Liberty SUV 2012, Jeep Patriot SUV 2012, Jeep Wrangler SUV 2012, Lamborghini Aventador Coupe 2012, Lamborghini Diablo Coupe 2001, Lamborghini Gallardo LP 570-4 Superleggera 2012, Lamborghini Reventon Coupe 2008, Land Rover LR2 SUV 2012, Land Rover Range Rover SUV 2012, Lincoln Town Car Sedan 2011, MINI Cooper Roadster Convertible 2012, Maybach Landaulet Convertible 2012, Mazda Tribute SUV 2011, McLaren MP4-12C Coupe 2012, Mercedes-Benz 300-Class Convertible 1993, Mercedes-Benz C-Class Sedan 2012, Mercedes-Benz E-Class Sedan 2012, Mercedes-Benz S-Class Sedan 2012, Mercedes-Benz SL-Class Coupe 2009, Mercedes-Benz Sprinter Van 2012, Mitsubishi Lancer Sedan 2012, Nissan 240SX Coupe 1998, Nissan Juke Hatchback 2012, Nissan Leaf Hatchback 2012, Nissan NV Passenger Van 2012, Plymouth Neon Coupe 1999, Porsche Panamera Sedan 2012, Ram C/V Cargo Van Minivan 2012, Rolls-Royce Ghost Sedan 2012, Rolls-Royce Phantom Drophead Coupe Convertible 2012, Rolls-Royce Phantom Sedan 2012, Scion xD Hatchback 2012, Spyker C8 Convertible 2009, Spyker C8 Coupe 2009, Suzuki Aerio Sedan 2007, Suzuki Kizashi Sedan 2012, Suzuki SX4 Hatchback 2012, Suzuki SX4 Sedan 2012, Tesla Model S Sedan 2012, Toyota 4Runner SUV 2012, Toyota Camry Sedan 2012, Toyota Corolla Sedan 2012, Toyota Sequoia SUV 2012, Volkswagen Beetle Hatchback 2012, Volkswagen Golf Hatchback 1991, Volkswagen Golf Hatchback 2012, Volvo 240 Sedan 1993, Volvo C30 Hatchback 2012, Volvo XC90 SUV 2007, smart fortwo Convertible 2012'
    data ={}
    data_samples_path=args.data_samples
    print(data_samples_path)
    images_dir=os.path.join(args.data_path)
    print(images_dir)
    
    with open(data_samples_path, 'r') as file:
        raw_data_labels = json.load(file)
    
    raw_data=os.listdir(images_dir)
    for sample in raw_data:
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

    
