import torch
import numpy as np
import folder_paths as comfy_paths
import comfy
from PIL import Image
import hashlib
import cv2
from typing import Tuple

BBox = Tuple[int, int, int, int]

models_dir =  comfy_paths.models_dir

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# PIL Hex
def pil2hex(image):
    return hashlib.sha256(np.array(tensor2pil(image)).astype(np.uint16).tobytes()).hexdigest()

# PIL to Mask
def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask

# Mask to PIL
def mask2pil(mask):
    if mask.ndim > 2:
        mask = mask.squeeze(0)
    mask_np = mask.cpu().numpy().astype('uint8')
    mask_pil = Image.fromarray(mask_np, mode="L")
    return mask_pil

# Tensor to cv2
def tensor2cv(image):
    image_np = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)      
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# cv2 to Tensor
def cv2tensor(image):
    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0    
    return torch.from_numpy(image_np).unsqueeze(0)

def hex2rgb(hex_color: str):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def hex2bgr(hex_color):
    return hex2rgb(hex_color)[::-1]
