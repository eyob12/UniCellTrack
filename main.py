

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from transformers import SamModel
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import base64
from PIL import Image
import io
from datasets import Dataset
from utils.process_util import post_processing
from dataset import ISICDataset,SAMDatasetWithPoints,processor
from  utils.prompt_util import point_prompt
from jupyter_bbox_widget import BBoxWidget
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from monai.losses import DiceLoss, DiceFocalLoss
import torch.nn as nn
# Step 5: Define Loss Functions and Optimizer
import matplotlib.pyplot as plt
from monai.losses import DiceLoss, DiceFocalLoss
from skimage.color import label2rgb
import torch
import torch.nn.functional as F

import cv2
import matplotlib.pyplot as plt
from monai.losses import DiceLoss, DiceFocalLoss
from skimage.color import label2rgb
import torch
import torch.nn.functional as F
import scipy
from scipy.ndimage import gaussian_filter, map_coordinates
# Function to encode image from a torch tensor to base64
def tensor_to_base64_preserve_detail(img_tensor):
    np_img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    if np_img.dtype != np.uint8:
        np_min, np_max = np_img.min(), np_img.max()
        np_img = ((np_img - np_min) / (np_max - np_min) * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(np_img)

    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')  # PNG is lossless
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

# List to store the clicked points
clicked_points = []
# Callback function for when a point is clicked on the image
def on_point_select(point):
    global clicked_points
    clicked_points.append((point[0], point[1]))  # store x, y coordinates
    print(f"Clicked at point: ({point[0]}, {point[1]})")


class Args:
    def __init__(self):
        
        self.dataroot ='./ALL_2D_CTC_Data_sep'
        self.SAM_Weight = '/root/Desktop/data/private/cell_seg_SAM/sam-vit-base'
        self.seed = 24
        self.gpu_ids = '0'
        self.port = '12355'
        self.exp_name = 'ALL_2D_CTC_Data_sep'
#Fluo_N2DH_SIM+
args = Args()


def rotate_90(image):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def rotate_180(image):
    return cv2.rotate(image, cv2.ROTATE_180)

def rotate_270(image):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Flip augmentations
def flip_horizontal(image):
    return cv2.flip(image, 1)

def flip_vertical(image):
    return cv2.flip(image, 0)

# def random_shift(image, max_shift=20):
#     h, w = image.shape[:2]
#     tx, ty = np.random.randint(-max_shift, max_shift + 1, size=2)
#     M = np.float32([[1, 0, tx], [0, 1, ty]])
#     return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def random_scale(image, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    h, w = image.shape[:2]
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def elastic_transform(image, alpha=34, sigma=4):
    random_state = np.random.RandomState(None)
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.array([
        (y + dy).flatten(),
        (x + dx).flatten()
    ])

    if image.ndim == 3:  # color or multi-channel
        transformed = np.zeros_like(image)
        for c in range(image.shape[2]):
            transformed[..., c] = map_coordinates(
                image[..., c], indices, order=1, mode='reflect'
            ).reshape(shape)
        return transformed
    else:
        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)



def random_brightness(image, factor_range=(0.8, 1.2)):
    factor = np.random.uniform(*factor_range)
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def random_contrast(image, factor_range=(0.8, 1.2)):
    factor = np.random.uniform(*factor_range)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

def add_gaussian_noise(image, sigma=10):
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

def random_blur(image, ksize=3):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def random_shift(image, max_shift=10):
    dx = np.random.randint(-max_shift, max_shift)
    dy = np.random.randint(-max_shift, max_shift)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

# augmentations = [
#     rotate_90, rotate_180, rotate_270,
#     flip_horizontal, flip_vertical,
#     random_shift, 
#     random_scale,
#    # elastic_transform,
#     #random_brightness, random_contrast,
#     add_gaussian_noise , random_blur
# ]

# augmentations = [rotate_90, rotate_180, rotate_270, flip_horizontal, 
#                  flip_vertical
#                   ]  

num_epochs=500

def mse_loss(pred, target):
    if pred.ndim == 2:  # If shape is (256, 256), add batch & channel dims
        pred = pred.unsqueeze(0).unsqueeze(0)  # Shape -> (1, 1, 256, 256)
        target = target.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:  # If shape is (B, 256, 256), add channel dim
        pred = pred.unsqueeze(1)  # Shape -> (B, 1, 256, 256)
        target = target.unsqueeze(1)

    return F.mse_loss(pred, target)

def gradient_difference_loss(pred, target):
    if pred.ndim == 2:  
        pred = pred.unsqueeze(0).unsqueeze(0)  # Ensure (B, C, H, W)
        target = target.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:  
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)

    if pred.shape[-2] < 2 or pred.shape[-1] < 2:
        raise ValueError(f"Invalid spatial dimensions: {pred.shape[-2:]}")

    dx_pred, dy_pred = torch.gradient(pred, dim=(-2, -1))
    dx_gt, dy_gt = torch.gradient(target, dim=(-2, -1))
    return F.l1_loss(dx_pred, dx_gt) + F.l1_loss(dy_pred, dy_gt)

def combined_loss(pred, target, lambda_1=1.0, lambda_2=0.5):
    return lambda_1 * mse_loss(pred, target) #+ lambda_2 * gradient_difference_loss(pred, target)



def random_gamma(image, gamma_range=(0.7, 1.5)):
    """Apply random gamma correction."""
    gamma = np.random.uniform(*gamma_range)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def random_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    img_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

def add_salt_pepper_noise(image, amount=0.01, salt_vs_pepper=0.5):
    """Add salt-and-pepper noise."""
    noisy = np.copy(image)
    num_pixels = int(amount * image.size)
    # Salt
    coords = [np.random.randint(0, i - 1, num_pixels) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    # Pepper
    coords = [np.random.randint(0, i - 1, num_pixels) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy

def random_motion_blur(image, kernel_size=5):
    """Apply random motion blur (linear)."""
    # Create kernel
    kernel = np.zeros((kernel_size, kernel_size))
    xs, ys = np.random.choice(kernel_size, 2, replace=False)
    if np.random.rand() > 0.5:
        kernel[xs, :] = 1.0 / kernel_size
    else:
        kernel[:, ys] = 1.0 / kernel_size
    return cv2.filter2D(image, -1, kernel)

def random_cutout(image, mask_size=32):
    """Randomly cut out a square patch (set to black)."""
    h, w = image.shape[:2]
    y = np.random.randint(0, h - mask_size)
    x = np.random.randint(0, w - mask_size)
    image_copy = image.copy()
    image_copy[y:y + mask_size, x:x + mask_size] = 0
    return image_copy

geo_augs = [
    rotate_90, rotate_180, rotate_270,
    flip_horizontal, flip_vertical,
    random_shift, random_scale
]

img_only_augs = [ add_gaussian_noise, random_motion_blur,
   # add_gaussian_noise,
   random_clahe, random_blur,random_brightness, random_contrast, random_gamma, add_salt_pepper_noise
]

if __name__ == '__main__':
    
    tfs=transforms.Compose([  
        transforms.ToTensor(),
        transforms.Resize((256, 256))   
    ])
    # dataset = ISICDataset(data_dir=args.dataroot,
    # transform=tfs,
    # augmentations=augmentations,
    # is_train=True)
#     dataset = ISICDataset(
#     data_dir=args.dataroot,
#     transform=tfs,
#     geo_augs=geo_augs,
#     img_only_augs=img_only_augs,
#     is_train=True
# )
    
    dataset = ISICDataset(
    data_dir=args.dataroot,
    transform=tfs,
    geo_augs=geo_augs,
    img_only_augs=img_only_augs,
    is_train=True,
    use_ssr_train=True,
    use_ssr_test=False,   # 
    ssr_sigma=60.0        # 
)
    print(f"whole_dataset after augmentations ={len(dataset)}")
    sam_dataset = SAMDatasetWithPoints(dataset, processor, is_train=True)
    train_dataloader = DataLoader(sam_dataset, batch_size=1, shuffle=True)
    #batch = next(iter(train_dataloader))



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the SAM model
    sam_checkpoint = args.SAM_Weight
    model = SamModel.from_pretrained(sam_checkpoint).to(device)

    # Unfreeze both the encoder and the mask decoder
    for name, param in model.named_parameters():
        param.requires_grad_(True)  # Unfreeze all parameters (encoder and mask decoder)
    # Load the trained checkpoint

        # Freeze all parameters
    # for param in model.parameters():
    #     param.requires_grad = False

    # # Unfreeze only the mask decoder
    # for name, param in model.mask_decoder.named_parameters():
    #     param.requires_grad = True

    #checkpoint_path = "/root/Desktop/data/private/CT_challenge/comb_datasets/BF-C2DL-HSC/models/Model_PhC-C2DL-PSC_02.pth"
    checkpoint_path = args.Trained_Weight
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # # Load model weights
    model.load_state_dict(checkpoint, strict=False)  # strict=False allows partial loading if necessary

    dice_loss_fn = DiceLoss(reduction='mean', jaccard=True, sigmoid=True, squared_pred=True).to(device)
    focal_loss_fn = DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean').to(device)
    criterion_distance = nn.SmoothL1Loss().to(device)  # For distance map

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)


    for epoch in range(num_epochs):
        epoch_losses = []
        count = 0


        for batch in tqdm(train_dataloader):

            pixel_values = batch["pixel_values"].to(device)  # Image tensor
            input_points = batch["input_points"].to(device)  # Points tensor
            ground_truth_mask = batch["ground_truth_mask"].to(device).unsqueeze(1)
            distance_map = batch["distance_map"].to(device).unsqueeze(1)

            # Ensure input_points has the correct shape [batch_size, point_batch_size, nb_points_per_image, 2]
            if len(input_points.shape) == 3:
                input_points = input_points.unsqueeze(2)  # Change shape from [1, 1, 2] to [1, 1, 1, 2]


            # Run the model on the current set of points
            outputs = model(
                pixel_values=pixel_values,
                input_points=input_points,  # [1, 1, nb_points_per_image, 2]
                multimask_output=True  # Output multiple masks if needed
            )

                # Extract predicted masks from the model output
            segmentation_output = outputs.pred_masks  # Shape: [1, N, 256, 256]
            

            predicted_masks_sum = torch.amax(segmentation_output, dim=(1,2)).squeeze()
            #predicted_masks_sum = torch.sigmoid(predicted_masks_sum).squeeze()  # Apply sigmoid for binary segmentation
            ground_truth_mask = ground_truth_mask.squeeze()
            distance_map=distance_map.squeeze()

            # distance_map = distance_map.squeeze()
            total_loss = combined_loss(predicted_masks_sum, distance_map)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_losses.append(total_loss.item())

            # count += 1
            # if count == 1:

            #     # Visualization
            #     image = pixel_values[0].cpu().detach().numpy().transpose(1, 2, 0)
            #     image = (image - image.min()) / (image.max() - image.min())
            #     image_uint8 = (image * 255).astype(np.uint8)
            #     ground_truth = ground_truth_mask.cpu().detach().numpy().squeeze()
            #     predicted_mask = predicted_masks_sum.cpu().detach().numpy().squeeze()
            #     distance_map = distance_map.cpu().detach().numpy().squeeze()

            #     # final_mask=post_processing(predicted_mask)
                

            #     input_points = batch["input_points"].squeeze(0).cpu().numpy()

            #     fig, axes = plt.subplots(1, 4, figsize=(35, 35))
            #     axes[0].imshow(image)
            #     axes[0].set_title('Image')
            #     axes[0].axis('off')
            #     axes[1].imshow(image)
            #     axes[1].scatter(input_points[:, 1], input_points[:, 0], c='red', s=50, marker='x', label='Point prompt')
            #     axes[1].set_title('Image with point')
            #     axes[1].axis('off')
            #     axes[1].legend()

            #     axes[2].imshow(ground_truth, cmap='gray')
            #     axes[2].set_title('Ground Truth Mask')
            #     axes[2].axis('off')


            #     axes[3].imshow(predicted_mask, cmap='gray')
            #     axes[3].set_title('Predicted Mask')
            #     axes[3].axis('off')
            #     plt.show()
            del ground_truth_mask, distance_map, input_points
            torch.cuda.empty_cache()

            # count += 1
            # if count == 5000:
            #     model_path = f"./{args.exp_name}.pth"
            #     torch.save(model.state_dict(), model_path)
            #     print(f"Training {epoch} completed and model saved at: {model_path}")
            #     count = 0


            # Epoch summary
        print(f"Epoch {epoch + 1}/{num_epochs}, Mean Loss: {np.mean(epoch_losses):.4f}")

        # Save Model

        model_path = f"./final_{args.exp_name}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Training {epoch} completed and model saved at: {model_path}")
    



