import hashlib
import logging
import os
from pathlib import Path
import numpy as np
import torch
from trimesh import Trimesh
import comfy.model_management as mm
from folder_paths import models_dir, get_filename_list, get_full_path_or_raise, get_folder_paths, get_save_image_path, get_output_directory
from modules.image_util import tensor2pil
from .hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from .hy3dgen.texgen import Hunyuan3DPaintPipeline
log = logging.getLogger(__name__)

model_dir = os.path.join(os.path.dirname(__file__), "models") 

def get_device_by_name(device):
    """
    "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"],{"default": "auto"}), 
    """
    if device == 'auto':
        try:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                # mps is not available on macos
                device = "cpu"
            elif torch.xpu.is_available():
                device = "xpu"
        except:
                raise AttributeError("What's your device(到底用什么设备跑的)？")
    print("\033[93mUse Device(使用设备):", device, "\033[0m")
    return device

class HunyuanImage23DModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (cls.hunyuan_model_list(), ),
                "fast": ("BOOLEAN", {"default": False})
            },
        }
    RETURN_TYPES = ("HunyunModel",)
    RETURN_NAMES = ("hunyuanmodel",)
    FUNCTION = "load_model"
    CATEGORY = "Hunyuan"
    
    @staticmethod
    def hunyuan_model_list():
        unet_models = get_filename_list("checkpoints")
        return list(filter(lambda x: os.path.basename(x).endswith(".safetensors"), unet_models)) + \
            list(filter(lambda x: os.path.basename(x).endswith('.ckpt'), unet_models))
    
    @classmethod
    def IS_CHANGED(s, model_path: str, fast: bool):
        m = hashlib.sha256()
        m.update(model_path.encode())
        m.update(str(fast).encode())
        return m.digest().hex()

    def load_model(self, model_path, fast):
        mm.unload_all_models()
        
        ckpt_path = get_full_path_or_raise("checkpoints", model_path)
        device = get_device_by_name('auto')
        extra_args = {}
        model_subdir = 'hunyuan3d-dit-v2-0'
        use_safetensors = model_path.endswith(".safetensors")
        if fast:
            extra_args['variant'] = 'fp16'
            model_subdir = 'hunyuan3d-dit-v2-0-fast'
        
        config_path = os.path.join(model_dir, model_subdir, "config.yaml")
        
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            ckpt_path,
            config_path,
            device=device,
            use_safetensors=use_safetensors,
            **extra_args
        )
        return (pipe,)

class Hunyuan3DPaintModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # "model_path": (cls.hunyuan_model_list(), ),
            },
        }
    RETURN_TYPES = ("HunyunInpaintModel",)
    RETURN_NAMES = ("hunyuaninpainmodel",)
    FUNCTION = "load_model"
    CATEGORY = "Hunyuan"
    
    @staticmethod
    def hunyuan_model_list():
        unet_models = get_filename_list("checkpoint")
        return list(filter(lambda x: os.path.basename(x).endswith(".safetensors"), unet_models)) + \
            list(filter(lambda x: os.path.basename(x).endswith('.ckpt'), unet_models))
    
    def load_model(self):
        mm.unload_all_models()
        model_path = get_folder_paths("models")
        hunyuan_path = os.path.join(model_path, "Hunyuan3D-2")
        return Hunyuan3DPaintPipeline.from_pretrained(hunyuan_path)
        
class HunyuanImage23DRunner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hunyuanmodel": ('HunyunModel',),
                "guidance_scale": ("FLOAT", {'default': 7.5, "min": 0, "max": 30, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 50, "min": 10, "max": 60, "step": 1}),
                "remove_floaters": ("BOOLEAN", {"default": True}),
                "remove_degenerate_faces": ("BOOLEAN", {"default": True}),
                "reduce_faces": ("BOOLEAN", {"default": True}),
                "max_facenum": ("INT", {"default": 40000, "min": 1, "max": 10000000, "step": 1}),
                "smooth_normals": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "run_pipeline"
    CATEGORY = "Hunyuan"
    
    @classmethod
    def IS_CHANGED(s,
                   image,
                   hunyuanmodel: Hunyuan3DDiTFlowMatchingPipeline,
                   guidance_scale,
                   num_inference_steps,
                   remove_floaters,
                   remove_degenerate_faces,
                   reduce_faces,
                   max_facenum,
                   smooth_normals
        ):
        m = hashlib.sha256()
        m.update(np.array(image).tobytes())
        m.update(str(hunyuanmodel.__hash__()).encode())
        m.update(str(guidance_scale).encode())
        m.update(str(num_inference_steps).encode())
        m.update(str(remove_floaters).encode())
        m.update(str(remove_degenerate_faces).encode())
        m.update(str(reduce_faces).encode())
        m.update(str(max_facenum).encode())
        m.update(str(smooth_normals).encode())
        return m.digest().hex()

    def run_pipeline(self,
                     image,
                   hunyuanmodel: Hunyuan3DDiTFlowMatchingPipeline,
                   guidance_scale,
                   num_inference_steps,
                   remove_floaters,
                   remove_degenerate_faces,
                   reduce_faces,
                   max_facenum,
                   smooth_normals
        ):
        if image.dim() == 2:
            image = torch.unsqueeze(image, 0)
        image = tensor2pil(image[0])
        mesh = hunyuanmodel(
            image=image,
            mc_algo='mc',
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.manual_seed(2025))[0]
        # mesh = FloaterRemover()(mesh)
        # mesh = DegenerateFaceRemover()(mesh)
        # mesh = FaceReducer()(mesh)
        new_mesh = mesh.copy()
        # mesh.export('mesh.glb')
        if remove_floaters:
            new_mesh = FloaterRemover()(new_mesh)
            log.info(f"Removed floaters, resulting in {new_mesh.vertices.shape[0]} vertices and {new_mesh.faces.shape[0]} faces")
        if remove_degenerate_faces:
            new_mesh = DegenerateFaceRemover()(new_mesh)
            log.info(f"Removed degenerate faces, resulting in {new_mesh.vertices.shape[0]} vertices and {new_mesh.faces.shape[0]} faces")
        if reduce_faces:
            new_mesh = FaceReducer()(new_mesh, max_facenum=max_facenum)
            log.info(f"Reduced faces, resulting in {new_mesh.vertices.shape[0]} vertices and {new_mesh.faces.shape[0]} faces")
        if smooth_normals:              
            new_mesh.vertex_normals = Trimesh.smoothing.get_vertices_normals(new_mesh)

        return (new_mesh, )
    
class Hunyuan3DMeshTextureRunner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "image": ("IMAGE",),
                "hunyuanmodel": ("HunyunInpaintModel",)
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "texture"
    CATEGORY = "Hunyuan"
    
    # @classmethod
    # def IS_CHANGED(s, mesh, image, hunyuaninpaintmodel: Hunyuan3DPaintPipeline):
    #     m = hashlib.sha256()
    #     m.update(np.array(mesh).tobytes())
    #     m.update(np.array(image).tobytes())
    #     m.update(str(hunyuaninpaintmodel.__hash__()).encode())
    #     return m.digest().hex()

    def texture(self, mesh, image, hunyuaninpaintmodel: Hunyuan3DPaintPipeline):
        mesh = hunyuaninpaintmodel(mesh, image=image)
        return mesh
    
        
class Hunyuan3DViewer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "hy3D"}),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
            },
            "optional": {
                "save_file": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "display"
    CATEGORY = "Hunyuan"

    def display(self, trimesh, filename_prefix, file_format, save_file):
        full_output_folder, filename, counter, subfolder, filename_prefix = get_save_image_path(filename_prefix, get_output_directory())
        output_glb_path = Path(full_output_folder, f'{filename}_{counter:05}_.{file_format}')
        output_glb_path.parent.mkdir(exist_ok=True)
        if save_file:
            trimesh.export(output_glb_path, file_type=file_format)
            filename = f'{filename}_{counter:05}_.{file_format}'
            relative_path = Path(subfolder) / filename
        else:
            filename = f'hy3dtemp_.{file_format}'
            temp_file = Path(full_output_folder, filename)
            trimesh.export(temp_file, file_type=file_format)
            relative_path = Path(subfolder) / filename
        saved = list()
        saved.append({
            "filename": filename,
            "type": "output",
            "subfolder": subfolder
        })
        return {"ui": {"mesh": saved}}
    
NODE_CLASS_MAPPINGS = {
    "HunyuanImage23DModelLoader": HunyuanImage23DModelLoader,
    "HunyuanImage23DRunner": HunyuanImage23DRunner,
    "Hunyuan3DViewer": Hunyuan3DViewer
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanImage23DModelLoader": "Load Hunyuan 3D Model",
    "HunyuanImage23DRunner": "Run Hunyuan Pipeline",
    "Hunyuan3DViewer": "Display Mesh"
}
