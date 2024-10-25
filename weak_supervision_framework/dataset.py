import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, image_names, root_dir, labels, transform=None):
        """
        Args:
            image_names (list): List of image file names.
            root_dir (string): Directory where images are stored.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_names = image_names
        self.root_dir = root_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Build the full path of the image file
        img_name = os.path.join(self.root_dir, self.image_names[idx])
        label = self.labels[idx]
        image = Image.open(img_name).convert('RGB')  # Load image as RGB

        # Apply any transformations (e.g., resize, normalization)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)


class HatefulMemesDataset(Dataset):
    def __init__(self, image_names, root_dir, labels, processor):
        """
        Args:
            data_frame (DataFrame): DataFrame containing image names and labels.
            image_dir (str): Directory where the images are stored.
            processor (CLIPProcessor): CLIP processor for preprocessing images.
        """
        self.image_names = image_names
        self.root_dir = root_dir
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get image name and label from the dataframe
        img_name = os.path.join(self.root_dir, self.image_names[idx])
        label = self.labels[idx]

        # Load and process image
        image = Image.open(img_name).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")

        # Return image and label
        return inputs['pixel_values'].squeeze(0), torch.tensor(label, dtype=torch.long)
