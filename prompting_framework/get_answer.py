import torch
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, CLIPModel 
from torch.nn.functional import cosine_similarity


def get_class_embeddings(prompts, tokenizer, text_encoder):
    """ 
        Function to get  class embeddings
    """
    text_inputs = tokenizer(prompts, padding="max_length", return_tensors="pt").to(device)
    outputs = text_encoder(**text_inputs)
    text_embedding = outputs.pooler_output
    return text_embedding
    
def get_query_embedding(query_prompt, tokenizer, text_encoder):
     """ 
        Function to get  single query embedding
    """
    query_input = tokenizer(query_prompt, padding="max_length", return_tensors="pt").to(device)
    query_output = text_encoder(**query_input)
    query_embedding = query_output.pooler_output
    return query_embedding

def compute_scores(class_embeddings, query_embedding, prompts):
     """ 
        Function to get compute and return best match amongst all classes
    """
     # Compute cosine similarity scores
    similarity_scores = cosine_similarity(query_embedding, class_embeddings, dim=1)  # Shape: [37]
    
    # Find the highest matching score and corresponding item
    max_score_index = torch.argmax(similarity_scores).item()
    max_score = similarity_scores[max_score_index].item()
    best_match = prompts[max_score_index]
    
    # Print the result
   # print(f"Best match: {best_match} with a similarity score of {max_score:.4f}")
    return best_match


"""
Example setup and Usage
"""
model_id = "openai/clip-vit-large-patch14"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = CLIPTokenizer.from_pretrained(model_id)
text_encoder = CLIPTextModel.from_pretrained(model_id).to(device)
model = CLIPModel.from_pretrained(model_id).to(device)

Eurosat_prompts = ["SeaLake", "PermanentCrop", "River", "Residential", "Pasture", "Industrial", "Highway", "HerbaceousVegetation", 
                   "Forest","AnnualCrop"]
query_prompt = ["A photo of a Sealake area"] 
query_embedding = get_query_embedding(query_prompt, tokenizer, text_encoder)
eurosat_embeddings = get_class_embeddings(Eurosat_prompts, tokenizer, text_encoder)
detected_class = compute_scores(eurosat_embeddings, query_embedding, Eurosat_prompts)