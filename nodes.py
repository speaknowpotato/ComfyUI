import torch

import os
import sys
import json
import hashlib
import traceback
import math
import time
import random

from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import numpy as np
import safetensors.torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


import comfy.utils


import comfy.model_management
from comfy.cli_args import args

import importlib

import folder_paths

def before_node_execution():
    comfy.model_management.throw_exception_if_processing_interrupted()

def interrupt_processing(value=True):
    comfy.model_management.interrupt_current_processing(value)

MAX_RESOLUTION=8192



class SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

class PreviewImage(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

class LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

class LoadImageMask:
    _color_channels = ["alpha", "red", "green", "blue"]
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True}),
                     "channel": (s._color_channels, ), }
                }

    CATEGORY = "mask"

    RETURN_TYPES = ("MASK",)
    FUNCTION = "load_image"
    def load_image(self, image, channel):
        image_path = folder_paths.get_annotated_filepath(image)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        if i.getbands() != ("R", "G", "B", "A"):
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            i = i.convert("RGBA")
        mask = None
        c = channel[0].upper()
        if c in i.getbands():
            mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
            if c == 'A':
                mask = 1. - mask
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (mask.unsqueeze(0),)

    @classmethod
    def IS_CHANGED(s, image, channel):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

class ImageScale:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                              "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                              "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                              "crop": (s.crop_methods,)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, image, upscale_method, width, height, crop):
        if width == 0 and height == 0:
            s = image
        else:
            samples = image.movedim(-1,1)

            if width == 0:
                width = max(1, round(samples.shape[3] * height / samples.shape[2]))
            elif height == 0:
                height = max(1, round(samples.shape[2] * width / samples.shape[3]))

            s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
            s = s.movedim(1,-1)
        return (s,)

class ImageScaleBy:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                              "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, image, upscale_method, scale_by):
        samples = image.movedim(-1,1)
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
        s = s.movedim(1,-1)
        return (s,)

class ImageInvert:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "invert"

    CATEGORY = "image"

    def invert(self, image):
        s = 1.0 - image
        return (s,)

class ImageBatch:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image1": ("IMAGE",), "image2": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch"

    CATEGORY = "image"

    def batch(self, image1, image2):
        if image1.shape[1:] != image2.shape[1:]:
            image2 = comfy.utils.common_upscale(image2.movedim(-1,1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1,-1)
        s = torch.cat((image1, image2), dim=0)
        return (s,)

class EmptyImage:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                              "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "color": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"

    CATEGORY = "image"

    def generate(self, width, height, batch_size=1, color=0):
        r = torch.full([batch_size, height, width, 1], ((color >> 16) & 0xFF) / 0xFF)
        g = torch.full([batch_size, height, width, 1], ((color >> 8) & 0xFF) / 0xFF)
        b = torch.full([batch_size, height, width, 1], ((color) & 0xFF) / 0xFF)
        return (torch.cat((r, g, b), dim=-1), )

class ImagePadForOutpaint:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "feathering": ("INT", {"default": 40, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "image"

    def expand_image(self, image, left, top, right, bottom, feathering):
        d1, d2, d3, d4 = image.size()

        new_image = torch.ones(
            (d1, d2 + top + bottom, d3 + left + right, d4),
            dtype=torch.float32,
        ) * 0.5

        new_image[:, top:top + d2, left:left + d3, :] = image

        mask = torch.ones(
            (d2 + top + bottom, d3 + left + right),
            dtype=torch.float32,
        )

        t = torch.zeros(
            (d2, d3),
            dtype=torch.float32
        )

        if feathering > 0 and feathering * 2 < d2 and feathering * 2 < d3:

            for i in range(d2):
                for j in range(d3):
                    dt = i if top != 0 else d2
                    db = d2 - i if bottom != 0 else d2

                    dl = j if left != 0 else d3
                    dr = d3 - j if right != 0 else d3

                    d = min(dt, db, dl, dr)

                    if d >= feathering:
                        continue

                    v = (feathering - d) / feathering

                    t[i, j] = v * v

        mask[top:top + d2, left:left + d3] = t

        return (new_image, mask)


NODE_CLASS_MAPPINGS = {
    "SaveImage": SaveImage,
    "PreviewImage": PreviewImage,
    "LoadImage": LoadImage,
    "LoadImageMask": LoadImageMask,
    "ImageScale": ImageScale,
    "ImageScaleBy": ImageScaleBy,
    "ImageInvert": ImageInvert,
    "ImageBatch": ImageBatch,
    "ImagePadForOutpaint": ImagePadForOutpaint,
    "EmptyImage": EmptyImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Image
    "SaveImage": "Save Image",
    "PreviewImage": "Preview Image",
    "LoadImage": "Load Image",
    "LoadImageMask": "Load Image (as Mask)",
    "ImageScale": "Upscale Image",
    "ImageScaleBy": "Upscale Image By",
    "ImageUpscaleWithModel": "Upscale Image (using Model)",
    "ImageInvert": "Invert Image",
    "ImagePadForOutpaint": "Pad Image for Outpainting",
    "ImageBatch": "Batch Images",
}

EXTENSION_WEB_DIRS = {}

def load_custom_node(module_path, ignore=set()):
    module_name = os.path.basename(module_path)
    if os.path.isfile(module_path):
        sp = os.path.splitext(module_path)
        module_name = sp[0]
    try:
        if os.path.isfile(module_path):
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
            module_dir = os.path.split(module_path)[0]
        else:
            module_spec = importlib.util.spec_from_file_location(module_name, os.path.join(module_path, "__init__.py"))
            module_dir = module_path

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)

        if hasattr(module, "WEB_DIRECTORY") and getattr(module, "WEB_DIRECTORY") is not None:
            web_dir = os.path.abspath(os.path.join(module_dir, getattr(module, "WEB_DIRECTORY")))
            if os.path.isdir(web_dir):
                EXTENSION_WEB_DIRS[module_name] = web_dir

        if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS") is not None:
            for name in module.NODE_CLASS_MAPPINGS:
                if name not in ignore:
                    NODE_CLASS_MAPPINGS[name] = module.NODE_CLASS_MAPPINGS[name]
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and getattr(module, "NODE_DISPLAY_NAME_MAPPINGS") is not None:
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            return True
        else:
            print(f"Skip {module_path} module for custom nodes due to the lack of NODE_CLASS_MAPPINGS.")
            return False
    except Exception as e:
        print(traceback.format_exc())
        print(f"Cannot import {module_path} module for custom nodes:", e)
        return False

def load_custom_nodes():
    base_node_names = set(NODE_CLASS_MAPPINGS.keys())
    node_paths = folder_paths.get_folder_paths("custom_nodes")
    node_import_times = []
    for custom_node_path in node_paths:
        possible_modules = os.listdir(os.path.realpath(custom_node_path))
        if "__pycache__" in possible_modules:
            possible_modules.remove("__pycache__")

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) and os.path.splitext(module_path)[1] != ".py": continue
            if module_path.endswith(".disabled"): continue
            time_before = time.perf_counter()
            success = load_custom_node(module_path, base_node_names)
            node_import_times.append((time.perf_counter() - time_before, module_path, success))

    if len(node_import_times) > 0:
        print("\nImport times for custom nodes:")
        for n in sorted(node_import_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (IMPORT FAILED)"
            print("{:6.1f} seconds{}:".format(n[0], import_message), n[1])
        print()

def init_custom_nodes():
    extras_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy_extras")
    extras_files = [
        "nodes_post_processing.py"
    ]

    for node_file in extras_files:
        load_custom_node(os.path.join(extras_dir, node_file))

    load_custom_nodes()
