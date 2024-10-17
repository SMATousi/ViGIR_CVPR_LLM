#Program to evaluate CLIP model on datasets
import os
import torch
from PIL import Image
from transformers  import CLIPProcessor, CLIPModel
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16

model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14-336",
        attn_implementation="flash_attention_2",
        device_map=device,
        torch_dtype=torch.float16,
        )

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

dataset_dir = "/home1/pupil/goowfd/CVPR_2025/hateful_memes/img/"

output_file = "/home1/pupil/goowfd/CVPR_2025/hateful_memes/CLIP_predictions_hateful_l14-dev_flash.json"

text_inputs = ["a photo of a hateful meme", "a photo of a non hateful meme"]


# Load the source JSON file that contains the image file names
source_json = "/home1/pupil/goowfd/CVPR_2025/hateful_memes/simplified_dev.json"
with open(source_json, "r") as f:
    image_data = json.load(f)

results = []
print("Running inference...")


# Iterate over the images provided in the source JSON file
for i, entry in enumerate(image_data):
    img_filename = entry["img"]
    img_path = os.path.join(dataset_dir, img_filename)

    print(f'Processing {img_filename} ({i+1} out of {len(image_data)})')

    # Check if the image file exists in the directory
    if os.path.exists(img_path):
        # Open and process the image
        image = Image.open(img_path)

        # Process the image and text inputs for the CLIP model
        inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
        inputs = inputs.to(device)

        # Run inference
        with torch.no_grad():
            with torch.autocast(device):
                outputs = model(**inputs)

        # Get the probabilities and predicted label
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        # 1 for hateful, 0 for non-hateful based on maximum probability
        predicted_label = 1 if probs.argmax(dim=1).item() == 0 else 0

        # Store the result (image filename and predicted label)
        results.append({"img": img_filename, "label": predicted_label})
    else:
        print(f"Image {img_filename} not found in dataset directory!")

# Save the results to a JSON file
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print("Done!")


