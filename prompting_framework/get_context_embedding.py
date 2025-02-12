import ollama 
import random
import numpy as np

def generate_context_embedding(class_names : str, model : str,  options : dict): -> list 
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