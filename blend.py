"""Implement Photoshop color blend
ref: https://stackoverflow.com/questions/12121393/please-explain-this-color-blending-mode-formula-so-i-can-replicate-it-in-php-ima
"""

from typing import Literal
import cv2
import numpy as np

R = 0.3
G = 0.59
B = 0.11

def rgb2hsy(image):
    """image is normalized to [0,1]
    """
    image = np.minimum(np.maximum(image, 0), 1)
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]

    h = np.zeros_like(r)
    s = np.zeros_like(g)
    y = R * r + G * g + B * b
    
    mask_gray = (r == g) & (g == b)
    s[mask_gray] = 0
    h[mask_gray] = 0

    mask_sector_0 = ((r >= g) & (g >= b)) & ~mask_gray # Sector 0: 0° - 60°
    s[mask_sector_0] = r[mask_sector_0] - b[mask_sector_0]
    h[mask_sector_0] = 60 * (g[mask_sector_0] - b[mask_sector_0]) / s[mask_sector_0]

    mask_sector_1 = (g > r) & (r >= b)
    s[mask_sector_1] = g[mask_sector_1] - b[mask_sector_1]
    h[mask_sector_1] = 60 * (g[mask_sector_1] - r[mask_sector_1]) / s[mask_sector_1]  + 60

    mask_sector_2 = (g >= b) & (b > r)
    s[mask_sector_2] = g[mask_sector_2] - r[mask_sector_2]
    h[mask_sector_2] = 60 * (b[mask_sector_2] - r[mask_sector_2]) / s[mask_sector_2] + 120

    mask_sector_3 = (b > g) & (g > r)
    s[mask_sector_3] = b[mask_sector_3] - r[mask_sector_3]
    h[mask_sector_3] = 60 * (b[mask_sector_3] - g[mask_sector_3]) / s[mask_sector_3] + 180

    mask_sector_4 = (b > r) & (r >= g)
    s[mask_sector_4] = b[mask_sector_4] - g[mask_sector_4]
    h[mask_sector_4] = 60 * (r[mask_sector_4] - g[mask_sector_4]) / s[mask_sector_4] + 240

    mask_sector_5 = ~(mask_gray | mask_sector_0 | mask_sector_1 | mask_sector_2 | mask_sector_3 | mask_sector_4)
    s[mask_sector_5] = r[mask_sector_5] - g[mask_sector_5]
    h[mask_sector_5] = 60 * (r[mask_sector_5] - b[mask_sector_5]) / s[mask_sector_5] + 300
    
    hsy = np.zeros_like(image)
    hsy[:,:,0] = h % 360
    hsy[:,:,1] = np.minimum(np.maximum(s, 0), 1)
    hsy[:,:,2] = np.minimum(np.maximum(y, 0), 1)
    return hsy
 
def hsy2rgb(image):
    h, s, y = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    h = h % 360
    s = np.minimum(np.maximum(s, 0), 1)
    y = np.minimum(np.maximum(y, 0), 1)
    
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    k = np.zeros_like(h)

    mask_sector_0 = (h >= 0) & (h < 60)
    k[mask_sector_0] = s[mask_sector_0] * h[mask_sector_0] / 60
    b[mask_sector_0] = y[mask_sector_0] - R * s[mask_sector_0] - G * k[mask_sector_0]
    r[mask_sector_0] = b[mask_sector_0] + s[mask_sector_0]
    g[mask_sector_0] = b[mask_sector_0] + k[mask_sector_0]

    mask_sector_1 = (h >= 60) & (h < 120)
    k[mask_sector_1] = s[mask_sector_1] * (h[mask_sector_1] - 60) / 60
    g[mask_sector_1] = y[mask_sector_1] + B * s[mask_sector_1] + R * k[mask_sector_1]
    b[mask_sector_1] = g[mask_sector_1] - s[mask_sector_1]
    r[mask_sector_1] = g[mask_sector_1] - k[mask_sector_1]

    mask_sector_2 = (h >= 120) & (h < 180)
    k[mask_sector_2] = s[mask_sector_2] * (h[mask_sector_2] - 120) / 60
    r[mask_sector_2] = y[mask_sector_2] - G * s[mask_sector_2] - B * k[mask_sector_2]
    g[mask_sector_2] = r[mask_sector_2] + s[mask_sector_2]
    b[mask_sector_2] = r[mask_sector_2] + k[mask_sector_2]

    mask_sector_3 = (h >= 180) & (h < 240)
    k[mask_sector_3] = s[mask_sector_3] * (h[mask_sector_3] - 180) / 60
    b[mask_sector_3] = y[mask_sector_3] + R * s[mask_sector_3] + G * k[mask_sector_3]
    r[mask_sector_3] = b[mask_sector_3] - s[mask_sector_3]
    g[mask_sector_3] = b[mask_sector_3] - k[mask_sector_3]

    mask_sector_4 = (h >= 240) & (h < 300)
    k[mask_sector_4] = s[mask_sector_4] * (h[mask_sector_4] - 240) / 60
    g[mask_sector_4] = y[mask_sector_4] - B * s[mask_sector_4] - R * k[mask_sector_4]
    b[mask_sector_4] = g[mask_sector_4] + s[mask_sector_4]
    r[mask_sector_4] = g[mask_sector_4] + k[mask_sector_4]

    mask_sector_5 = h >= 300
    k[mask_sector_5] = s[mask_sector_5] * (h[mask_sector_5] - 300) / 60
    r[mask_sector_5] = y[mask_sector_5] + G * s[mask_sector_5] + B * k[mask_sector_5]
    g[mask_sector_5] = r[mask_sector_5] - s[mask_sector_5]
    b[mask_sector_5] = r[mask_sector_5] - k[mask_sector_5]
            
    return np.minimum(np.maximum(np.stack([r, g, b], axis=-1), 0), 1)


def color_blend(base_image, blend_image, mode: Literal["Hue", "Saturation", "Color", "Luminosity"]):
    """
    Args:
        base_image (cv2)
        blend_image (cv2)
    Return:
        cv2
    """
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    blend_image = cv2.cvtColor(blend_image, cv2.COLOR_BGR2RGB).astype(np.float32)/255

    # Convert to HLS color space using OpenCV
    hsy_base = rgb2hsy(base_image)
    hsy_blend = rgb2hsy(blend_image)

    if mode=="Hue":
        hsy_out = np.stack([hsy_blend[:,:,0], hsy_base[:,:,1], hsy_base[:,:,2]], axis=-1)
    elif mode=="Saturation":
        hsy_out = np.stack([hsy_base[:,:,0], hsy_blend[:,:,1], hsy_base[:,:,2]], axis=-1)
    elif mode=="Color":
        hsy_out = np.stack([hsy_blend[:,:,0], hsy_blend[:,:,1], hsy_base[:,:,2]], axis=-1)
    elif mode=="Luminosity":
        hsy_out = np.stack([hsy_base[:,:,0], hsy_base[:,:,1], hsy_blend[:,:,2]], axis=-1)
    else: assert False, f"{mode} is not a valid mode"
    rgb_out = hsy2rgb(hsy_out)
    return cv2.cvtColor((rgb_out*255).astype(np.uint8), cv2.COLOR_RGB2BGR)