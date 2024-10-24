import ollama 
import os
from tqdm import tqdm
import json
import signal
import argparse
import wandb

def main():

    parser = argparse.ArgumentParser(description="A script to run V-LLMs on different image classification datasets")
        
    # Add an argument for an integer option
    parser.add_argument("--runname", type=str, required=False, help="The wandb run name.")
    parser.add_argument("--projectname", type=str, required=False, help="The wandb project name.")
    parser.add_argument("--wandbapi", type=str, required=False, help="The wandb private API key to login.")
    parser.add_argument("--modelname", type=str, required=True, help="The name of the V-LLM model")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt that you want to give to the V-LLM")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the image data dir")
    parser.add_argument("--results_path", type=str, required=True, help="Path to the results .json file (must include .json)")
    parser.add_argument("--timeout", type=int, default=20, help="time out duration to skip one sample")
    # parser.add_argument("--savingstep", type=int, default=100)
    # parser.add_argument("--epochs", type=int, default=100)
    # parser.add_argument("--imagesize", type=int, default=256)
    # parser.add_argument("--alpha", type=float, default=1)
    # parser.add_argument("--beta", type=float, default=1)
    # parser.add_argument("--threshold", type=float, default=1)
    # parser.add_argument("--dropoutrate", type=float, default=0.5)
    parser.add_argument("--debug", action="store_true", help="Enables debugging mode. It will run the pipeline just on one sample.")
    parser.add_argument("--logging", action="store_true", help="Enables logging to the wandb")
    parser.add_argument("--dev", action="store_true", help="Enables evaluation on dev set.")
    # parser.add_argument("--train", action="store_true", help="Enables training on the train set.")
    parser.add_argument("--model_unloading", action="store_true", help="Enables unloading mode. Every 100 sampels it unloades the model from the GPU to avoid carshing.")

    args = parser.parse_args()



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



    # root_path = '/home1/pupil/goowfd/CVPR_2025/hateful_memes/img'
    root_path = args.data_path

    if args.dev:
        with open('/root/home/data/hateful_memes/simplified_dev.json', 'r') as file:
            data = json.load(file)

        list_of_image_names = []
        for entry in data:
            img_name, ext = entry['img'].split('.')
            padded_img_name = img_name.zfill(5)  # Pad the image name to 5 digits
            list_of_image_names.append(f"{padded_img_name}.{ext}")
        # print(image_names)
    else:
        list_of_image_names = os.listdir(root_path)

    

    # print(image_names)

    # model_name = 'llava:7b'
    model_name = args.modelname

    # results_file_name = model_name + '_results_hateful.json'
    results_file_name = args.results_path

    ollama.pull(model_name)

    timeout_duration = args.timeout

    print(f"Handling the timeout exceptions with timeout duration of {timeout_duration} seconds")

    options= {  # new
                "seed": 123,
                "temperature": 0,
                "num_ctx": 2048, # must be set, otherwise slightly random output
            }
        
    llava_7b_labels = {}
    count = 0

    for image_name in tqdm(list_of_image_names):
        
        count = count + 1

        image_path = os.path.join(root_path, image_name)
        
        prompt = args.prompt
        # prompt = "Is this an offensive meme? Please answer with YES or NO. DO NOT mention the reason: "
    #     prompt = "Is there ? Please answer with YES or NO. DO NOT mention the reason: "
    #     prompt = "Is this somehow an offensive meme? Please answer with YES or NO: "
    #     prompt = "describe this image: "
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_duration)  # Set the timeout

        try:
            if args.model_unloading:
                if count % 99 == 0:
                    response = ollama.generate(model=model_name, prompt=prompt, images=[image_path], options=options, keep_alive=0)
                else:
                    response = ollama.generate(model=model_name, prompt=prompt, images=[image_path], options=options)
            else:
                response = ollama.generate(model=model_name, prompt=prompt, images=[image_path], options=options)
            
            label = check_yes_no(response['response'])

        except TimeoutException:
            print(f"Prompt for {image_name} took longer than {timeout_duration} seconds. Moving to the next one.")
            label = None

        finally:
            signal.alarm(0)  # Disable the alarm

        
        
        
        llava_7b_labels[image_name] = label
        

        if args.debug:
            break

    with open(results_file_name, 'w') as fp:
        json.dump(llava_7b_labels, fp)


    if args.logging:

        wandb.login(key=args.wandbapi) 

        wandb.init(
                        # set the wandb project where this run will be logged
                    project=args.projectname, name=args.runname
                        
                        # track hyperparameters and run metadata
                        # config={
                        # "learning_rate": 0.02,
                        # "architecture": "CNN",
                        # "dataset": "CIFAR-100",
                        # "epochs": 20,
                        # }
                )
        
        wandb.log(llava_7b_labels)

if __name__ == "__main__":
    main()