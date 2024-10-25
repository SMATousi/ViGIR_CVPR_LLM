from snorkel.labeling import labeling_function
import json
import os
import numpy as np
from snorkel.labeling import LFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import LabelModel
from snorkel.analysis import metric_score
from snorkel.utils import probs_to_preds
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import argparse
import wandb
from dataset import *
from model import *
from LFs import *
from utils import *


def main():

    parser = argparse.ArgumentParser(description="A script to run V-LLMs on different image classification datasets")
        
    # Add an argument for an integer option
    parser.add_argument("--runname", type=str, default=None, required=False, help="The wandb run name.")
    parser.add_argument("--projectname", type=str, default=None, required=False, help="The wandb project name.")
    parser.add_argument("--modelname", type=str, required=True, help="The name of the model")
    parser.add_argument("--epochs", type=int, default=100, help="The number of epochs")
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    parser.add_argument("--debug", action="store_true", help="Enables debugging mode. It will run the pipeline just on one sample.")
    parser.add_argument("--logging", action="store_true", help="Enables logging to the wandb")

    args = parser.parse_args()


    train_data_json_path = '/home/macula/SMATousi/CVPR/ViGIR_CVPR_LLM/prompting_framework/prompting_results/hateful/simplified_train.json'
    dev_data_json_path = '/home/macula/SMATousi/CVPR/ViGIR_CVPR_LLM/prompting_framework/prompting_results/hateful/simplified_dev.json'

    with open(train_data_json_path, 'r') as file:
        train_data = json.load(file)

    # Extract and pad image names, ensuring they are 5 digits long before the '.png'
    train_image_names = []
    for entry in train_data:
        img_name, ext = entry['img'].split('.')
        padded_img_name = img_name.zfill(5)  # Pad the image name to 5 digits
        train_image_names.append(f"{padded_img_name}.{ext}")

    with open(dev_data_json_path, 'r') as file:
        dev_data = json.load(file)
        
    dev_image_names = []
    Y_dev = []
    for entry in dev_data:
        Y_dev.append(entry['label'])
        img_name, ext = entry['img'].split('.')
        padded_img_name = img_name.zfill(5)  # Pad the image name to 5 digits
        dev_image_names.append(f"{padded_img_name}.{ext}")

    print(f"There are {len(train_image_names)} images in the Train set.")
    print(f"There are {len(dev_image_names)} images in the dev set.")
    print(f"There are {len(Y_dev)} labels in the dev set.")


    lfs = [llava_7b,
       moondream,
       llava_llama3,
       llava_phi3,
       bakllava
       ]

    applier = LFApplier(lfs)

    L_dev = applier.apply(dev_image_names)
    L_train = applier.apply(train_image_names)
    Y_dev = np.array(Y_dev)

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train, Y_dev, n_epochs=5000, log_freq=500, seed=12345)


    probs_dev = label_model.predict_proba(L_dev)
    preds_dev = probs_to_preds(probs_dev)
    print(
        f"Label model f1 score: {metric_score(Y_dev, preds_dev, probs=probs_dev, metric='f1')}"
    )
    print(
        f"Label model roc-auc: {metric_score(Y_dev, preds_dev, probs=probs_dev, metric='roc_auc')}"
    )

    root_dir = "/home1/pupil/goowfd/CVPR_2025/hateful_memes/img/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    # Create datasets and dataloaders
    if args.modelname == "CLIP":
        train_dataset = HatefulMemesDataset(image_names=train_image_names, root_dir=root_dir, labels=preds_train, processor=processor)
        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=16)

        dev_dataset = HatefulMemesDataset(image_names=dev_image_names, root_dir=root_dir, labels=Y_dev, processor=processor)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch, shuffle=False)

    if args.modelname == "ResNet":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224 (example)
            transforms.ToTensor(),          # Convert images to PyTorch tensors
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean/std
        ])

        train_dataset = ImageDataset(image_names=train_image_names, root_dir=root_dir, labels=preds_train, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=16)

        dev_dataset = ImageDataset(image_names=dev_image_names, root_dir=root_dir, labels=Y_dev, transform=transform)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch, shuffle=False, num_workers=16)

    # Define MLP head (the dimension is based on CLIP output size)
    mlp_head = MLPHead(input_dim=512, output_dim=2)  # Binary classification, so output_dim = 2

    # Create the full model with CLIP + MLP
    if args.modelname == "CLIP":
        model = CLIPWithMLP(clip_model=clip_model, mlp_head=mlp_head)
    if args.modelname == "ResNet":
        model = ResNetWithMLP(num_classes=2)
    

    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.modelname == "CLIP":
        optimizer = optim.Adam(model.mlp_head.parameters(), lr=0.0001)
    if args.modelname == "ResNet":
        optimizer = optim.Adam(model.resnet.fc.parameters(), lr=0.0001)

    # Train the model
    epochs = 100
    train_model(model, 
                train_loader, 
                dev_loader,
                criterion, 
                optimizer, 
                device,
                logging=args.logging,
                debug=args.debug,
                project_name=args.projectname,
                run_name=args.runname, 
                epochs=args.epochs)

    # Evaluate the model
    evaluate_model(model, dev_loader, device, args.debug)



if __name__ == "__main__":
    main()