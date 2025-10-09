from __future__ import annotations
import torch


import os
import sys
import json
import hashlib
import inspect
import traceback
import math
import time
import random
import logging

from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo

import numpy as np
import safetensors.torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict, FileLocator
from comfy_api.internal import register_versions, ComfyAPIWithVersion
from comfy_api.version_list import supported_versions
from comfy_api.latest import io, ComfyExtension

import comfy.clip_vision

import comfy.model_management
from comfy.cli_args import args

import importlib

import folder_paths
import latent_preview
import node_helpers

def before_node_execution():
    comfy.model_management.throw_exception_if_processing_interrupted()

def interrupt_processing(value=True):
    comfy.model_management.interrupt_current_processing(value)

MAX_RESOLUTION=16384


# Import all core nodes from modular comfy_core files
from comfy_core import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Import individual node classes for backwards compatibility (so comfy_extras can inherit from them)
from comfy_core.core_image import SaveImage, LoadImage, PreviewImage, LoadImageMask, LoadImageOutput, ImageScale, ImageScaleBy, ImageInvert, ImageBatch, EmptyImage, ImagePadForOutpaint
from comfy_core.core_controlnet import ControlNetApply, ControlNetApplyAdvanced
from comfy_core.core_latent import VAEDecode, VAEEncode, VAEDecodeTiled, VAEEncodeTiled, VAEEncodeForInpaint, EmptyLatentImage, LatentUpscale, LatentUpscaleBy, LatentFromBatch, RepeatLatentBatch, LatentRotate, LatentFlip, LatentComposite, LatentBlend, LatentCrop, SetLatentNoiseMask, SaveLatent, LoadLatent
from comfy_core.core_conditioning import CLIPTextEncode, ConditioningCombine, ConditioningAverage, ConditioningConcat, ConditioningSetArea, ConditioningSetAreaPercentage, ConditioningSetAreaStrength, ConditioningSetMask, ConditioningZeroOut, ConditioningSetTimestepRange, InpaintModelConditioning
from comfy_core.core_loaders import CheckpointLoader, CheckpointLoaderSimple, DiffusersLoader, unCLIPCheckpointLoader, CLIPSetLastLayer, LoraLoader, LoraLoaderModelOnly, VAELoader, ControlNetLoader, DiffControlNetLoader, UNETLoader, CLIPLoader, DualCLIPLoader, CLIPVisionLoader, StyleModelLoader, GLIGENLoader
from comfy_core.core_sampling import KSampler, KSamplerAdvanced
from comfy_core.core_clip import CLIPVisionEncode, unCLIPConditioning
from comfy_core.core_style import StyleModelApply
from comfy_core.core_gligen import GLIGENTextBoxApply

EXTENSION_WEB_DIRS = {}

# Dictionary of successfully loaded module names and associated directories.
LOADED_MODULE_DIRS = {}


def get_module_name(module_path: str) -> str:
    """
    Returns the module name based on the given module path.
    Examples:
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node.py") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node/") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node/__init__.py") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node/__init__") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node/__init__/") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node.disabled") -> "custom_nodes
    Args:
        module_path (str): The path of the module.
    Returns:
        str: The module name.
    """
    base_path = os.path.basename(module_path)
    if os.path.isfile(module_path):
        base_path = os.path.splitext(base_path)[0]
    return base_path


async def load_custom_node(module_path: str, ignore=set(), module_parent="custom_nodes") -> bool:
    module_name = get_module_name(module_path)
    if os.path.isfile(module_path):
        sp = os.path.splitext(module_path)
        module_name = sp[0]
        sys_module_name = module_name
    elif os.path.isdir(module_path):
        sys_module_name = module_path.replace(".", "_x_")

    try:
        logging.debug("Trying to load custom node {}".format(module_path))
        if os.path.isfile(module_path):
            module_spec = importlib.util.spec_from_file_location(sys_module_name, module_path)
            module_dir = os.path.split(module_path)[0]
        else:
            module_spec = importlib.util.spec_from_file_location(sys_module_name, os.path.join(module_path, "__init__.py"))
            module_dir = module_path

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[sys_module_name] = module
        module_spec.loader.exec_module(module)

        LOADED_MODULE_DIRS[module_name] = os.path.abspath(module_dir)

        try:
            from comfy_config import config_parser

            project_config = config_parser.extract_node_configuration(module_path)

            web_dir_name = project_config.tool_comfy.web

            if web_dir_name:
                web_dir_path = os.path.join(module_path, web_dir_name)

                if os.path.isdir(web_dir_path):
                    project_name = project_config.project.name

                    EXTENSION_WEB_DIRS[project_name] = web_dir_path

                    logging.info("Automatically register web folder {} for {}".format(web_dir_name, project_name))
        except Exception as e:
            logging.warning(f"Unable to parse pyproject.toml due to lack dependency pydantic-settings, please run 'pip install -r requirements.txt': {e}")

        if hasattr(module, "WEB_DIRECTORY") and getattr(module, "WEB_DIRECTORY") is not None:
            web_dir = os.path.abspath(os.path.join(module_dir, getattr(module, "WEB_DIRECTORY")))
            if os.path.isdir(web_dir):
                EXTENSION_WEB_DIRS[module_name] = web_dir

        # V1 node definition
        if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS") is not None:
            for name, node_cls in module.NODE_CLASS_MAPPINGS.items():
                if name not in ignore:
                    NODE_CLASS_MAPPINGS[name] = node_cls
                    node_cls.RELATIVE_PYTHON_MODULE = "{}.{}".format(module_parent, get_module_name(module_path))
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and getattr(module, "NODE_DISPLAY_NAME_MAPPINGS") is not None:
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            return True
        # V3 Extension Definition
        elif hasattr(module, "comfy_entrypoint"):
            entrypoint = getattr(module, "comfy_entrypoint")
            if not callable(entrypoint):
                logging.warning(f"comfy_entrypoint in {module_path} is not callable, skipping.")
                return False
            try:
                if inspect.iscoroutinefunction(entrypoint):
                    extension = await entrypoint()
                else:
                    extension = entrypoint()
                if not isinstance(extension, ComfyExtension):
                    logging.warning(f"comfy_entrypoint in {module_path} did not return a ComfyExtension, skipping.")
                    return False
                node_list = await extension.get_node_list()
                if not isinstance(node_list, list):
                    logging.warning(f"comfy_entrypoint in {module_path} did not return a list of nodes, skipping.")
                    return False
                for node_cls in node_list:
                    node_cls: io.ComfyNode
                    schema = node_cls.GET_SCHEMA()
                    if schema.node_id not in ignore:
                        NODE_CLASS_MAPPINGS[schema.node_id] = node_cls
                        node_cls.RELATIVE_PYTHON_MODULE = "{}.{}".format(module_parent, get_module_name(module_path))
                    if schema.display_name is not None:
                        NODE_DISPLAY_NAME_MAPPINGS[schema.node_id] = schema.display_name
                return True
            except Exception as e:
                logging.warning(f"Error while calling comfy_entrypoint in {module_path}: {e}")
                return False
        else:
            logging.warning(f"Skip {module_path} module for custom nodes due to the lack of NODE_CLASS_MAPPINGS or NODES_LIST (need one).")
            return False
    except Exception as e:
        logging.warning(traceback.format_exc())
        logging.warning(f"Cannot import {module_path} module for custom nodes: {e}")
        return False

async def init_external_custom_nodes():
    """
    Initializes the external custom nodes.

    This function loads custom nodes from the specified folder paths and imports them into the application.
    It measures the import times for each custom node and logs the results.

    Returns:
        None
    """
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
            if args.disable_all_custom_nodes and possible_module not in args.whitelist_custom_nodes:
                logging.info(f"Skipping {possible_module} due to disable_all_custom_nodes and whitelist_custom_nodes")
                continue
            time_before = time.perf_counter()
            success = await load_custom_node(module_path, base_node_names, module_parent="custom_nodes")
            node_import_times.append((time.perf_counter() - time_before, module_path, success))

    if len(node_import_times) > 0:
        logging.info("\nImport times for custom nodes:")
        for n in sorted(node_import_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (IMPORT FAILED)"
            logging.info("{:6.1f} seconds{}: {}".format(n[0], import_message, n[1]))
        logging.info("")

async def init_builtin_extra_nodes():
    """
    Initializes the built-in extra nodes in ComfyUI.

    This function loads the extra node files located in the "comfy_extras" directory and imports them into ComfyUI.
    If any of the extra node files fail to import, a warning message is logged.

    Returns:
        None
    """
    extras_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy_extras")
    extras_files = [
        "nodes_latent.py",
        "nodes_hypernetwork.py",
        "nodes_upscale_model.py",
        "nodes_post_processing.py",
        "nodes_mask.py",
        "nodes_compositing.py",
        "nodes_rebatch.py",
        "nodes_model_merging.py",
        "nodes_tomesd.py",
        "nodes_clip_sdxl.py",
        "nodes_canny.py",
        "nodes_freelunch.py",
        "nodes_custom_sampler.py",
        "nodes_hypertile.py",
        "nodes_model_advanced.py",
        "nodes_model_downscale.py",
        "nodes_images.py",
        "nodes_video_model.py",
        "nodes_train.py",
        "nodes_sag.py",
        "nodes_perpneg.py",
        "nodes_stable3d.py",
        "nodes_sdupscale.py",
        "nodes_photomaker.py",
        "nodes_pixart.py",
        "nodes_cond.py",
        "nodes_morphology.py",
        "nodes_stable_cascade.py",
        "nodes_differential_diffusion.py",
        "nodes_ip2p.py",
        "nodes_model_merging_model_specific.py",
        "nodes_pag.py",
        "nodes_align_your_steps.py",
        "nodes_attention_multiply.py",
        "nodes_advanced_samplers.py",
        "nodes_webcam.py",
        "nodes_audio.py",
        "nodes_sd3.py",
        "nodes_gits.py",
        "nodes_controlnet.py",
        "nodes_hunyuan.py",
        "nodes_eps.py",
        "nodes_flux.py",
        "nodes_lora_extract.py",
        "nodes_torch_compile.py",
        "nodes_mochi.py",
        "nodes_slg.py",
        "nodes_mahiro.py",
        "nodes_lt.py",
        "nodes_hooks.py",
        "nodes_load_3d.py",
        "nodes_cosmos.py",
        "nodes_video.py",
        "nodes_lumina2.py",
        "nodes_wan.py",
        "nodes_lotus.py",
        "nodes_hunyuan3d.py",
        "nodes_primitive.py",
        "nodes_cfg.py",
        "nodes_optimalsteps.py",
        "nodes_hidream.py",
        "nodes_fresca.py",
        "nodes_apg.py",
        "nodes_preview_any.py",
        "nodes_ace.py",
        "nodes_string.py",
        "nodes_camera_trajectory.py",
        "nodes_edit_model.py",
        "nodes_tcfg.py",
        "nodes_context_windows.py",
        "nodes_qwen.py",
        "nodes_chroma_radiance.py",
        "nodes_model_patch.py",
        "nodes_easycache.py",
        "nodes_audio_encoder.py",
    ]

    import_failed = []
    for node_file in extras_files:
        if not await load_custom_node(os.path.join(extras_dir, node_file), module_parent="comfy_extras"):
            import_failed.append(node_file)

    return import_failed


async def init_builtin_api_nodes():
    api_nodes_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy_api_nodes")
    api_nodes_files = [
        "nodes_ideogram.py",
        "nodes_openai.py",
        "nodes_minimax.py",
        "nodes_veo2.py",
        "nodes_kling.py",
        "nodes_bfl.py",
        "nodes_bytedance.py",
        "nodes_luma.py",
        "nodes_recraft.py",
        "nodes_firefly.py",
        "nodes_pixverse.py",
        "nodes_stability.py",
        "nodes_pika.py",
        "nodes_runway.py",
        "nodes_sora.py",
        "nodes_tripo.py",
        "nodes_moonvalley.py",
        "nodes_rodin.py",
        "nodes_gemini.py",
        "nodes_vidu.py",
        "nodes_wan.py",
    ]

    if not await load_custom_node(os.path.join(api_nodes_dir, "canary.py"), module_parent="comfy_api_nodes"):
        return api_nodes_files

    import_failed = []
    for node_file in api_nodes_files:
        if not await load_custom_node(os.path.join(api_nodes_dir, node_file), module_parent="comfy_api_nodes"):
            import_failed.append(node_file)

    return import_failed

async def init_public_apis():
    register_versions([
        ComfyAPIWithVersion(
            version=getattr(v, "VERSION"),
            api_class=v
        ) for v in supported_versions
    ])

async def init_extra_nodes(init_custom_nodes=True, init_api_nodes=True):
    await init_public_apis()

    import_failed = await init_builtin_extra_nodes()

    import_failed_api = []
    if init_api_nodes:
        import_failed_api = await init_builtin_api_nodes()

    if init_custom_nodes:
        await init_external_custom_nodes()
    else:
        logging.info("Skipping loading of custom nodes")

    if len(import_failed_api) > 0:
        logging.warning("WARNING: some comfy_api_nodes/ nodes did not import correctly. This may be because they are missing some dependencies.\n")
        for node in import_failed_api:
            logging.warning("IMPORT FAILED: {}".format(node))
        logging.warning("\nThis issue might be caused by new missing dependencies added the last time you updated ComfyUI.")
        if args.windows_standalone_build:
            logging.warning("Please run the update script: update/update_comfyui.bat")
        else:
            logging.warning("Please do a: pip install -r requirements.txt")
        logging.warning("")

    if len(import_failed) > 0:
        logging.warning("WARNING: some comfy_extras/ nodes did not import correctly. This may be because they are missing some dependencies.\n")
        for node in import_failed:
            logging.warning("IMPORT FAILED: {}".format(node))
        logging.warning("\nThis issue might be caused by new missing dependencies added the last time you updated ComfyUI.")
        if args.windows_standalone_build:
            logging.warning("Please run the update script: update/update_comfyui.bat")
        else:
            logging.warning("Please do a: pip install -r requirements.txt")
        logging.warning("")

    return import_failed
