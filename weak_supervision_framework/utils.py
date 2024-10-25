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
import wandb


# Training function
def train_model(model, 
                train_loader, 
                dev_loader,
                criterion, 
                optimizer, 
                device,
                logging=False,
                debug=False,
                project_name=None,
                run_name=None, 
                epochs=5):

    
    if logging:

        wandb.init(
                        # set the wandb project where this run will be logged
                    project=project_name, name=run_name
                        
                        # track hyperparameters and run metadata
                        # config={
                        # "learning_rate": 0.02,
                        # "architecture": "CNN",
                        # "dataset": "CIFAR-100",
                        # "epochs": 20,
                        # }
                )

    # Set model to training mode
    precision, recall, f1 = evaluate_model(model, dev_loader, device, debug)
    if logging:
        wandb.log({"Train/Loss": 0, 
                    "Dev/precision": precision,
                    "Dev/recall": recall,
                    "Dev/F1": f1})

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if debug:
                break

        precision, recall, f1 = evaluate_model(model, dev_loader, device, debug)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

        if logging:
            wandb.log({"Train/Loss": running_loss / len(train_loader), 
                        "Dev/precision": precision,
                        "Dev/recall": recall,
                        "Dev/F1": f1})

# Evaluation function to compute precision, recall, and F1-score
def evaluate_model(model, 
                dev_loader, 
                device,
                debug=False):
                
    model.eval()  # Set model to evaluation mode
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(dev_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            if debug:
                break

    # Compute metrics
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return precision, recall, f1