
from .ISIC_dataset1 import * 
from .SAMPointDataset import *



def processor(image, input_points=None, input_boxes=None, return_tensors="pt"):
    """
    Custom processor function for SAM that handles images, point prompts, and bounding boxes.
    
    Args:
        image (np.ndarray or PIL.Image.Image): The input image.
        input_points (torch.Tensor): Tensor of points with shape (N, 2) where N is the number of points.
        input_boxes (torch.Tensor): Tensor of bounding boxes with shape (N, 4) where N is the number of boxes.
        return_tensors (str): Return type for the processed data ("pt" for PyTorch, "np" for NumPy).
    
    Returns:
        dict: A dictionary containing processed image, points, and bounding boxes.
    """
    # Define transformations for preprocessing the image
    image_size = 1024
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    
    # Preprocess image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    pixel_values = transform(image)

    # # Process input points (normalize to image size)
    # if input_points is not None:
    #     # Ensure input_points is 2D with shape [N, 2]
    #     if input_points.dim() > 2:
    #         input_points = input_points[:, :2]  # Keep only x, y coordinates

    #     # Normalize points to [0, 1] based on image dimensions
    #     #input_points = input_points / torch.tensor([image.width, image.height], dtype=torch.float32)  # Normalize
    # Process input points (scale to the new image size)
    if input_points is not None:
        # Ensure input_points is 2D with shape [N, 2]
        if input_points.dim() > 2:
            input_points = input_points[:, :2]  # Keep only x, y coordinates

        # Scaling factor to convert from 256x256 to 1024x1024
        scale_factor = torch.tensor([1024 / 256, 1024 / 256], dtype=torch.float32)

        # Scale the points to the new image size (1024x1024)
        # print(input_points.shape,scale_factor) [23,3] [4,4]
        input_points = input_points * scale_factor  # Scale each point
        
        # Optionally: Normalize points to [0, 1] based on the new image dimensions
        #input_points = input_points / torch.tensor([1024, 1024], dtype=torch.float32)  # Normalize to [0, 1]

    # Process input boxes (normalize to image size)
    if input_boxes is not None:
        input_boxes[:, [0, 2]] = input_boxes[:, [0, 2]] / image.width  # Normalize x-coordinates
        input_boxes[:, [1, 3]] = input_boxes[:, [1, 3]] / image.height  # Normalize y-coordinates

    # Prepare output
    result = {"pixel_values": pixel_values.unsqueeze(0)}  # Add batch dimension
    if input_points is not None:
        result["input_points"] = input_points.unsqueeze(0)  # Add batch dimension
    if input_boxes is not None:
        result["input_boxes"] = input_boxes.unsqueeze(0)  # Add batch dimension
    
    if return_tensors == "np":
        result = {key: value.numpy() for key, value in result.items()}

    return result