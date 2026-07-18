import os 
import cv2
import torch
import skimage.io as io
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from skimage import exposure

# ---------------- SSR PREPROCESSOR ----------------
def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx == mn:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - mn) / (mx - mn)
    return (img * 255.0 + 0.5).astype(np.uint8)

def single_scale_retinex(image_rgb_or_gray: np.ndarray, sigma: float = 60.0) -> np.ndarray:
    x = _to_uint8(image_rgb_or_gray).astype(np.float32) / 255.0
    if x.ndim == 2:
        x = x[..., None]

    blurred = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    ssr = np.log(x + 1e-8) - np.log(blurred + 1e-8)

    H, W, C = ssr.shape
    ssr_flat = ssr.reshape(-1, C)
    p1 = np.percentile(ssr_flat, 1, axis=0)
    p99 = np.percentile(ssr_flat, 99, axis=0)
    denom = (p99 - p1); denom[denom == 0] = 1.0
    ssr = (ssr - p1) / denom
    ssr = np.clip(ssr, 0.0, 1.0)

    out = (ssr * 255.0 + 0.5).astype(np.uint8)
    if out.shape[2] == 1:
        out = out[..., 0]
    return out

# ---------------- IMAGE CROPPER ----------------
def _crop_with_roi_box(img: np.ndarray, y1: int, y2: int, x1: int, x2: int) -> np.ndarray:
    return img[y1:y2, x1:x2]

# --------- Per-dataset crop settings (test time only) ----------
CROP_BOXES = {
    # "BF-C2DL-MuSC":(80, 970, 70, 1000) ,
    "BF-C2DL-MuSC":(80, 970, 70, 1000) ,  # y1, y2, x1, x2 01-(70, 980, 70, 1010) 01-(20, 960, 70, 1010) 02-tra (80, 970, 70, 1000)
    # "BF-C2DL-HSC": (150, 940, 30, 970),  #test 01-[500:980, 200:970] #02-[250:970, 300:970]  02-tra (150, 940, 30, 300)
    # Add more datasets as needed
}
# ---------------------------------------------------------------

class ISICDataset(Dataset):
    def __init__(
        self,
        data_dir,
        dataset_name=None,
        transform=None,
        geo_augs=None,
        img_only_augs=None,
        is_train=False,
        use_ssr_train: bool = False,
        use_ssr_test: bool = True,
        ssr_sigma: float = 60.0,
    ):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.transform = transform
        self.geo_augs = geo_augs or []
        self.img_only_augs = img_only_augs or []
        self.is_train = is_train
        self.use_ssr_train = use_ssr_train
        self.use_ssr_test = use_ssr_test
        self.ssr_sigma = ssr_sigma

        self.crop_box = CROP_BOXES.get(dataset_name, None)

        if self.is_train:
            self.images_dir = os.path.join(data_dir, "processed_image") #processed_image, distance_map, processed_mask
            self.masks_dir = os.path.join(data_dir, "processed_mask")#images, masks
            self.marker_dir = os.path.join(data_dir, "distance_map")
            self.image_names = sorted(os.listdir(self.images_dir))
            self.mask_names = sorted(os.listdir(self.masks_dir))
            self.marker_names = sorted(os.listdir(self.marker_dir))
        else:
            self.images_dir = os.path.join(data_dir)
            self.image_names = sorted(os.listdir(self.images_dir))

        self.augmented_indices = []
        for idx in range(len(self.image_names)):
            self.augmented_indices.append((idx, None, None))
            for geo_idx in range(len(self.geo_augs)):
                self.augmented_indices.append((idx, geo_idx, None))
            for img_aug_idx in range(len(self.img_only_augs)):
                self.augmented_indices.append((idx, None, img_aug_idx))

    def __len__(self):
        return len(self.augmented_indices)

    def _load_rgb(self, path: str) -> np.ndarray:
        image = io.imread(path)

        if (image.dtype != np.uint8) or (image.min() < 0) or (image.max() > 255):
            image = exposure.rescale_intensity(
                image, in_range=(image.min(), image.max()), out_range=(0, 255)
            ).astype(np.uint8)

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] > 3:
            image = image[:, :, :3]
        elif image.shape[-1] < 3:
            while image.shape[-1] < 3:
                image = np.concatenate([image, image[:, :, :1]], axis=-1)

        if (not self.is_train) and (self.crop_box is not None):
            y1, y2, x1, x2 = self.crop_box
            image = _crop_with_roi_box(image, y1, y2, x1, x2)

        return image

    def __getitem__(self, idx):
        orig_idx, geo_idx, img_aug_idx = self.augmented_indices[idx]
        image_path = os.path.join(self.images_dir, self.image_names[orig_idx])
        image = self._load_rgb(image_path)

        if self.is_train:
            mask = io.imread(os.path.join(self.masks_dir, self.mask_names[orig_idx]))
            marker = io.imread(os.path.join(self.marker_dir, self.marker_names[orig_idx]))

            if geo_idx is not None:
                aug_fn = self.geo_augs[geo_idx]
                image = aug_fn(image)
                mask = aug_fn(mask)
                marker = aug_fn(marker)

            if img_aug_idx is not None:
                aug_fn = self.img_only_augs[img_aug_idx]
                image = aug_fn(image)

            if self.use_ssr_train:
                image = single_scale_retinex(image, sigma=self.ssr_sigma)
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            if self.transform:
                image_rgb = self.transform(image)
                mask = self.transform(mask)
                marker = self.transform(marker)
            else:
                image_rgb = image

            return {"image": image_rgb, "label": mask, "distance_map": marker}

        else:
            if geo_idx is not None:
                image = self.geo_augs[geo_idx](image)
            if img_aug_idx is not None:
                image = self.img_only_augs[img_aug_idx](image)

            if self.use_ssr_test:
                image = single_scale_retinex(image, sigma=self.ssr_sigma)
                
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            if self.transform:
                image_rgb = self.transform(image)
            else:
                image_rgb = image

            return {
                "image": image_rgb,
                "crop_box": self.crop_box,
                "image_name": self.image_names[orig_idx],
            }
