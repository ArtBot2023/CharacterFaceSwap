from typing import Tuple, List, Literal
import os
import comfy.samplers
from facexlib.detection import RetinaFace
from facexlib.parsing import BiSeNet
import torch
import numpy as np
import cv2
from .utils import models_dir, tensor2pil, pil2tensor, tensor2cv, cv2tensor, hex2bgr, BBox

# copy from SeargeSDXL
class GenerationParameterInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "image_width": ("INT", {"default": 512, "min": 0, "max": 1024, "step": 8}),
                    "image_height": ("INT", {"default": 512, "min": 0, "max": 1024, "step": 8}),
                    "steps": ("INT", {"default": 20, "min": 0, "max": 200}),
                    "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m"}),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras"}),
                    "denoise": ("FLOAT", {"default": 1, "min": 0, "max": 1})
                    },
                "optional": {
                    "parameters": ("PARAMETERS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETERS", )
    RETURN_NAMES = ("parameters", )
    FUNCTION = "mux"

    CATEGORY = "CFaceSwap"

    def mux(self, seed, image_width, image_height, steps, cfg, sampler_name, scheduler, denoise, parameters={}):
        parameters["seed"] = seed
        parameters["image_width"] = image_width
        parameters["image_height"] = image_height
        parameters["steps"] = steps
        parameters["cfg"] = cfg
        parameters["sampler_name"] = sampler_name
        parameters["scheduler"] = scheduler
        parameters["denoise"] = denoise
        return (parameters, )

class GenertaionParameterOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "parameters": ("PARAMETERS", ),
                    },
                }

    RETURN_TYPES = ("PARAMETERS", "INT", "INT", "INT", "INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "FLOAT", )
    RETURN_NAMES = ("parameters", "seed", "image_width", "image_height", "steps", "cfg", "sampler_name", "scheduler", "denoise", )
    FUNCTION = "demux"

    CATEGORY = "CFaceSwap"

    def demux(self, parameters):
        seed = parameters["seed"]
        image_width = parameters["image_width"]
        image_height = parameters["image_height"]
        steps = parameters["steps"]
        cfg = parameters["cfg"]
        sampler_name = parameters["sampler_name"]
        scheduler = parameters["scheduler"]
        denoise = parameters["denoise"]
        return (parameters, seed, image_width, image_height, steps, cfg, sampler_name, scheduler, denoise, )


class LoadRetinaFace:
    models_dir = os.path.join(models_dir, 'facexlib')
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{}}
    
    RETURN_TYPES = ("RETINAFACE", )
    RETURN_NAMES = ("MODEL", )
    FUNCTION = "load"
    CATEGORY = "CFaceSwap"
    def load(self):
        from facexlib.detection import init_detection_model
        return (init_detection_model("retinaface_resnet50", model_rootpath=self.models_dir), )
    
    
class CropFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("RETINAFACE", ),
                "image": ("IMAGE", ),
                "confidence": ("FLOAT", {"default": 0.8, "min": 0, "max": 1}),
                "margin": ("INT", {"default": 32}),
            }
        }
    
    RETURN_TYPES = (
        "IMAGE", "IMAGE", "BBOX"
    )
    RETURN_NAMES = ("face_image", "preview", "bbox")
    FUNCTION = "crop"
    CATEGORY = "CFaceSwap"

    def crop(self, model: RetinaFace, image: torch.Tensor, confidence: float, margin: int):        
        with torch.no_grad():
            # model receives bgr uint8 format
            # bboxes: list of [x0, y0, x1, y1, confidence_score, five points (x, y)]
            bboxes = model.detect_faces(tensor2cv(image), confidence)
        if (len(bboxes)==0):
            print("no face detected")
            return np.zeros((4,)), 0, image
        detection_preview = self.visualize_detection(tensor2cv(image), bboxes)
        bboxes = [self.add_margin_and_make_square((int(min(x0, x1)), int(min(y0,y1)), int(abs(x1-x0)), int(abs(y1-y0))), margin, img_width=image.shape[2], img_height=image.shape[1]) for (x0, y0, x1, y1, *_) in bboxes]
        detection_preview = self.visualize_margin(detection_preview, bboxes)

        faces = self.crop_faces(bboxes, image)
        # scaled_faces = self.scale_faces(faces, size)
        return faces[0].unsqueeze(0), cv2tensor(detection_preview), bboxes[0]
    
    def crop_faces(self, bboxes: List[BBox], image: torch.Tensor):
        """
        Returns: list of Tensor[h,w,c]
        """
        return [image[0, y:y+h, x:x+w, :] for (x,y,w,h) in bboxes]
    
    def scale_faces(self, faces: List[torch.Tensor], size: int, upscaler: Literal["linear"]="linear"):
        """
        Args:
            faces: list of Tensor[h,w,c]
        """
        scaled_faces: List[torch.Tensor] = []
        for face in faces:
            # Change the layout to [batch, channel, height, width]
            face = face.permute(2, 0, 1).unsqueeze(0)
            
            # Perform the interpolation
            if upscaler == "linear":
                scaled_face = torch.nn.functional.interpolate(face, size=(size, size), mode="bilinear", align_corners=True)
            elif upscaler == "nearest":
                scaled_face = torch.nn.functional.interpolate(face, size=(size, size), mode="nearest")
            else:
                raise ValueError(f"Invalid upscaler: {upscaler}")
            
            # Change the layout back to [height, width, channel] and remove batch dimension
            scaled_face = scaled_face.squeeze(0).permute(1, 2, 0)
            scaled_faces.append(scaled_face)
        
        return scaled_faces

    def visualize_margin(self, img, bboxes):
        img = np.copy(img)
        for bbox in bboxes:
            x,y,w,h = bbox
            cv2.rectangle(img, (x,y), (x+w, y+h), hex2bgr("#710193"), 2)
        return img

    def visualize_detection(self, img, bboxes_and_landmarks):
        """
        Args:
            img (np.ndarray): bgr
        Returns:
            img: bgr
        """
        img = np.copy(img)

        for b in bboxes_and_landmarks:
            # confidence
            cv2.putText(img, f'{b[4]:.4f}', (int(b[0]), int(b[1] + 12)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            # bounding boxes
            b = list(map(int, b))
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            # landmarks (for retinaface)
            cv2.circle(img, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img, (b[13], b[14]), 1, (255, 0, 0), 4)
        return img    
    
    def add_margin_and_make_square(self, bbox: BBox, margin: int, img_width: int, img_height: int):
        x, y, w, h = map(lambda x: int(x), bbox)  # x, y are the coordinates of the top-left corner of the bounding box
        
        # Calculate margin
        margin_w = margin
        margin_h = margin
        
        # Add margin to the bounding box, ensuring it doesn't go out of the image boundaries
        x = max(0, x - margin_w)
        y = max(0, y - margin_h)
        w = min(img_width - x, w + 2 * margin_w)
        h= min(img_height - y, h + 2 * margin_h)
        
        # Make the bounding box square while keeping the center the same
        cx, cy = x + w // 2, y + h // 2  # Calculate the center of the original bounding box
        max_side = max(w, h)
        x = max(0, cx - max_side // 2)
        y = max(0, cy - max_side // 2)
        w = h = min(max_side, img_width - x, img_height - y)  # Ensure the bounding box is within the image boundaries
        return int(x), int(y), int(w), int(h)

class UncropFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "bbox": ("BBOX", ),
                "face": ("IMAGE", ),
                "mask": ("MASK", )
            }
        }
    RETURN_TYPES = ("IMAGE", )
    
    CATEGORY = "CFaceSwap"
    FUNCTION = "uncrop"
    def uncrop(self, image: torch.Tensor, bbox: BBox, face: torch.Tensor, mask: torch.Tensor):
        bbox_face, bbox_mask = self.scale_face(face.squeeze(), mask, bbox[2])
        image_apply_face = self.weighted_sum(image.squeeze(), bbox, bbox_face, bbox_mask)
        return (image_apply_face.unsqueeze(0), )

    def scale_face(self, face: torch.Tensor, mask: torch.Tensor, size):
        """
        Args:
            face (torch.Tensor): [h,w,c]
            mask (torch.Tensor): [h,w]
        """
        scaled_faces: List[torch.Tensor] = []
        for face in [face, mask.unsqueeze(-1)]:
            print(f"face dims {face.shape}")
            # Change the layout to [batch, channel, height, width]
            face = face.permute(2, 0, 1).unsqueeze(0)            
            scaled_face = torch.nn.functional.interpolate(face, size=(size, size), mode="bilinear", align_corners=True)
            
            # Change the layout back to [height, width, channel] and remove batch dimension
            scaled_face = scaled_face.squeeze(0).permute(1, 2, 0)
            scaled_faces.append(scaled_face)
        return scaled_faces[0], scaled_faces[1].squeeze()

    def weighted_sum(self, image: torch.Tensor, bbox: BBox, face: torch.Tensor, mask: torch.Tensor)->torch.Tensor:
        """
        Args:
            image (torch.Tensor): [h_full,w_full,c]
            bbox (BBox): [x,y,w,h]
            face (torch.Tensor): [h,w,c]
            mask (torch.Tensor): [h,w]

        Returns:
            torch.Tensor: same shape as image
        """
        image = image.clone()
        x,y,w,h = bbox
        mask = mask.unsqueeze(-1)
        image[y:y+h, x:x+w, :] = mask * face + image[y:y+h, x:x+w, :] * (1-mask)
        return image

class LoadBisenet:
    models_dir = os.path.join(models_dir, 'facexlib')
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{}}
    
    RETURN_TYPES = ("BISENET", )
    FUNCTION = "load"
    CATEGORY = "CFaceSwap"
    def load(self):
        from facexlib.parsing import init_parsing_model
        return (init_parsing_model("bisenet", model_rootpath=self.models_dir), )

class SegFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BISENET", ),
                "image": ("IMAGE", ),
                "expand": ("INT", {"min": 0}),
                "include_hair": (["enable", "disable"], {"default": "disable"}),
                "include_neck": (["enable", "disable"], {"default": "disable"}),
            }
        }
    
    RETURN_TYPES = (
        "IMAGE", "MASK"
    )
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "segment"
    CATEGORY = "CFaceSwap"

    # labels: 0 'background'
    # 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye',
    # 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose',
    # 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l',
    # 16 'cloth', 17 'hair', 18 'hat'
    annotation_name = ['background', 
                  'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
                'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose',
                'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l',
                'cloth', 'hair', 'hat']
    annotation_color = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 85], [255, 0, 170], [0, 255, 0], [85, 255, 0],
                    [170, 255, 0], [0, 255, 85], [0, 255, 170], [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255],
                    [0, 170, 255], [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255],
                    [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    
    def segment(self, model: BiSeNet, image: torch.Tensor, expand, include_hair, include_neck):   
        image = image.squeeze().permute(2,0,1).unsqueeze(0).flip([1]) # shape [1, c, h, w], rgb2bgr
        with torch.no_grad():
            from torchvision.transforms.functional import normalize
            out = model(normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).cuda())[0] # shape [1, 19, h, w]  
        annotation = out.squeeze().cpu().numpy().argmax(0)
        mask = self.get_mask(annotation, expand, include_hair, include_neck)
        masked_face = mask.unsqueeze(0).unsqueeze(-1) * image.squeeze().permute(1,2,0).unsqueeze(0).flip([3]) # shape [1, h, w, c], bgr2rgb
        return masked_face, mask

    def get_mask(self, annotation, expand, include_hair, include_neck):
        face_inds = list(range(1,14))
        hair_ind = self.annotation_name.index("hair")
        neck_ind = self.annotation_name.index("neck")
        target_inds = face_inds
        if include_hair == "enable": target_inds.append(hair_ind)
        if include_neck == "enable": target_inds.append(neck_ind)

        mask = np.zeros_like(annotation, dtype=np.float32)
        for ind in target_inds:
            mask[annotation==ind] = 1
        blurred_mask = cv2.dilate(mask, np.ones((expand, expand), np.float32), iterations=1)
        return torch.from_numpy(blurred_mask) 

class ImageFullBBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", )
            }
        }
    
    RETURN_TYPES = ("BBOX", )
    FUNCTION = "bbox"
    CATEGORY = "CFaceSwap"
    def bbox(self, image: torch.Tensor):
        image = image.squeeze()
        return ((0,0,image.shape[1],image.shape[0]), )

class ColorBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blend_image": ("IMAGE", ),
                "base_image": ("IMAGE", ),
                "mode": (["Hue", "Saturation", "Color", "Luminosity"], )
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "blend"
    CATEGORY = "CFaceSwap"
    def blend(self, blend_image: torch.Tensor, base_image: torch.Tensor, mode: Literal["Hue", "Saturation", "Color", "Luminosity"]):
        from .blend import color_blend
        return (cv2tensor(color_blend(base_image=tensor2cv(base_image), blend_image=tensor2cv(blend_image), mode=mode)), )

class ExcludeFacialFeature:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face": ("IMAGE", ),
                "model": ("BISENET", ),
                "image": ("IMAGE", ),
                "expand": ("INT", {"min": 0})
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "exclude"
    CATEGORY = "CFaceSwap"
    annotation_name = ['background', 
        'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
        'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose',
        'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l',
        'cloth', 'hair', 'hat']
    
    def exclude(self, face: torch.Tensor, model: BiSeNet, image: torch.Tensor, expand: int):
        face = face.squeeze().permute(2,0,1).unsqueeze(0).flip([1]) # shape [1, c, h, w], rgb2bgr
        with torch.no_grad():
            from torchvision.transforms.functional import normalize
            out = model(normalize(face, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).cuda())[0] # shape [1, 19, h, w]  
        annotation = out.squeeze().cpu().numpy().argmax(0)
        # mask = self.get_mask_facial_feature(annotation, expand)
        # image = image * (1-mask).unsqueeze(0).unsqueeze(-1)
        mask = self.get_mask_hair(annotation, expand)
        image = image * mask.unsqueeze(0).unsqueeze(-1)
        return (image, )

    def get_mask_hair(self, annotation, expand):
        hair_ind = self.annotation_name.index('hair')
        mask = np.zeros_like(annotation, dtype=np.float32)
        mask[annotation==hair_ind] = 1
        blurred_mask = cv2.dilate(mask, np.ones((expand, expand), np.float32), iterations=1)
        return torch.from_numpy(blurred_mask)
    
    def get_mask_facial_feature(self, annotation, expand):
        facial_feature_inds = list(range(2,14))
        mask = np.zeros_like(annotation, dtype=np.float32)
        for ind in facial_feature_inds:
            mask[annotation==ind] = 1
        blurred_mask = cv2.dilate(mask, np.ones((expand, expand), np.float32), iterations=1)
        return torch.from_numpy(blurred_mask) 

class MaskContour:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", ),
            }
        }
    
    RETURN_TYPES = ("MASK", )
    FUNCTION = "find_contour"
    CATEGORY = "CFaceSwap"
    
    def find_contour(self, mask: torch.Tensor):
        mask_np: np.ndarray = mask.squeeze().cpu().numpy().astype('uint8')
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_contour = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.float32)
        cv2.drawContours(mask_contour, contours, -1, (1, ), 1)
        
        return (torch.from_numpy(mask_contour), )


NODE_CLASS_MAPPINGS = {
    "Generation Parameter Input": GenerationParameterInput,
    "Generation Parameter Output": GenertaionParameterOutput,
    "Load RetinaFace": LoadRetinaFace,
    "Load BiseNet": LoadBisenet,
    "Uncrop Face": UncropFace,
    "Crop Face": CropFace,
    "Segment Face": SegFace,
    "Image Full BBox": ImageFullBBox,
    "Color Blend": ColorBlend,
    "Exclude Facial Feature": ExcludeFacialFeature,
    "Mask Contour": MaskContour
}