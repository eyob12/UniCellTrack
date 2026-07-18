import torch
import numpy as np
from PIL import Image 
from torch.utils.data import Dataset
import torchvision.transforms as T
from utils.prompt_util import point_prompt

# Preprocess image
def preprocess_image(image, target_size=(1024, 1024)):
    if isinstance(image, np.ndarray):
        if len(image.shape) > 3:
            image = np.squeeze(image)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        elif len(image.shape) == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        elif len(image.shape) == 3 and image.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {image.shape}.")

        image = Image.fromarray(image)
        image = image.resize(target_size, Image.BICUBIC)
    return image

class SAMDatasetWithPoints(Dataset):
    def __init__(self, dataset, processor, target_size=(1024, 1024), is_train=True):
        self.dataset = dataset
        self.processor = processor
        self.target_size = target_size
        self.is_train = is_train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = np.array(sample["image"])
        image = preprocess_image(image, target_size=self.target_size)

        if self.is_train:
            distance_map = np.array(sample["distance_map"])
            
            distance_map= distance_map.squeeze(0)  # Ensure it's 2D
            #print(distance_map.shape, "sam dataset distance map")
            ground_truth_mask = np.array(sample["label"])
            centroids = point_prompt(distance_map)
            #points = torch.tensor(centroids, dtype=torch.float32)
            points = torch.tensor(np.array(centroids), dtype=torch.float32)


            #print(points.shape, "points.shape")

            inputs = self.processor(image=image, input_points=points, return_tensors="pt")

            return {
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "input_points": inputs["input_points"].squeeze(0),
                "ground_truth_mask": torch.tensor(ground_truth_mask, dtype=torch.float32).squeeze(0),
                "distance_map": torch.tensor(distance_map, dtype=torch.float32).squeeze(0),
            }
        else:
            inputs = self.processor(image=image, return_tensors="pt")
            return {
                "pixel_values": inputs["pixel_values"].squeeze(0),
            }
