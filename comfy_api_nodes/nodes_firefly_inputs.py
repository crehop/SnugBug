"""
Adobe Firefly Input Nodes for ComfyUI

Specialized input nodes for Firefly API parameters, allowing modular workflow construction.
Each parameter can use the built-in widget in the main node OR connect to these specialized input nodes.
"""

from __future__ import annotations
from inspect import cleandoc
from comfy.comfy_types.node_typing import IO
from comfy_api_nodes.apinode_utils import validate_string
from comfy_api_nodes.apis.firefly_api import FireflyContentClass


# Supported aspect ratios for Firefly API
FIREFLY_ASPECT_RATIOS = [
    "2048x2048 (1:1)",
    "2688x1536 (16:9)",
    "2304x1792 (4:3)",
    "1792x2304 (3:4)",
    "1344x768 (16:9)",
    "1152x896 (4:3)",
    "1024x1024 (1:1)",
    "896x1152 (3:4)",
    "2688x3456 (9:16)",
    "3456x2688 (16:9)",
]


# ============================================================================
# Text Input Nodes
# ============================================================================

class PromptInputNode:
    """
    Main prompt input with character validation.
    """

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("prompt",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/inputs/text"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Main generation prompt (max 1024 characters)"
                }),
            },
        }

    def execute(self, prompt: str = ""):
        validate_string(prompt, strip_whitespace=False, max_length=1024)
        return (prompt,)


class NegativePromptInputNode:
    """
    Negative prompt input for exclusions.
    """

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("negative_prompt",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/inputs/text"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "negative_prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Negative prompt to exclude unwanted elements"
                }),
            },
        }

    def execute(self, negative_prompt: str = ""):
        return (negative_prompt,)


class PromptSuffixInputNode:
    """
    Prompt suffix to append to main prompt.
    """

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("prompt_suffix",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/inputs/text"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "suffix": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text to append to main prompt"
                }),
            },
        }

    def execute(self, suffix: str = ""):
        return (suffix,)


class CustomModelIdInputNode:
    """
    Custom model ID input.
    """

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("custom_model_id",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/inputs/text"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "custom_model_id": (IO.STRING, {
                    "default": "",
                    "tooltip": "Custom model identifier (for *_custom model versions)"
                }),
            },
        }

    def execute(self, custom_model_id: str = ""):
        return (custom_model_id,)


class PromptBiasingLocaleInputNode:
    """
    Locale code for prompt biasing.
    """

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("locale_code",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/inputs/text"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "locale_code": (IO.STRING, {
                    "default": "",
                    "tooltip": "Language/locale code (e.g., 'en-US', 'ja-JP')"
                }),
            },
        }

    def execute(self, locale_code: str = ""):
        return (locale_code,)


class StylePresetInputNode:
    """
    Style preset selector.
    """

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("style_preset",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/inputs/enum"

    @classmethod
    def INPUT_TYPES(s):
        # Import from main file
        from comfy_api_nodes.nodes_firefly import FIREFLY_STYLE_PRESETS
        return {
            "optional": {
                "style_preset": (
                    FIREFLY_STYLE_PRESETS,
                    {"default": "none"}
                ),
            },
        }

    def execute(self, style_preset: str = "none"):
        return (style_preset,)


class UploadIdInputNode:
    """
    Generic upload ID input (for style/structure references).
    """

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("upload_id",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/inputs/text"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "upload_id": (IO.STRING, {
                    "default": "",
                    "tooltip": "Firefly storage upload ID"
                }),
            },
        }

    def execute(self, upload_id: str = ""):
        return (upload_id,)


# ============================================================================
# Enum/Dropdown Input Nodes
# ============================================================================

class ContentClassInputNode:
    """
    Content class selector (photo/art).
    """

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("content_class",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/inputs/enum"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "content_class": (
                    [c.value for c in FireflyContentClass],
                    {"default": FireflyContentClass.PHOTO.value}
                ),
            },
        }

    def execute(self, content_class: str = "photo"):
        return (content_class,)


class SizeInputNode:
    """
    Size/aspect ratio selector.
    """

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("size",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/inputs/enum"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "size": (
                    FIREFLY_ASPECT_RATIOS,
                    {"default": "2048x2048 (1:1)"}
                ),
            },
        }

    def execute(self, size: str = "2048x2048 (1:1)"):
        return (size,)


class ModelVersionInputNode:
    """
    Model version selector.
    """

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("model_version",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/inputs/enum"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_version": (
                    ["image3", "image3_custom", "image4_standard",
                     "image4_ultra", "image4_custom"],
                    {"default": "image4_standard"}
                ),
            },
        }

    def execute(self, model_version: str = "image4_standard"):
        return (model_version,)


class UpsamplerTypeInputNode:
    """
    Upsampler type selector.
    """

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("upsampler_type",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/inputs/enum"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "upsampler_type": (
                    ["default", "low_creativity"],
                    {"default": "default"}
                ),
            },
        }

    def execute(self, upsampler_type: str = "default"):
        return (upsampler_type,)


# ============================================================================
# Integer Input Nodes
# ============================================================================

class NumVariationsInputNode:
    """
    Number of variations selector.
    """

    RETURN_TYPES = (IO.INT,)
    RETURN_NAMES = ("num_variations",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/inputs/integer"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_variations": (IO.INT, {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "tooltip": "Number of variations to generate"
                }),
            },
        }

    def execute(self, num_variations: int = 1):
        return (num_variations,)


class SeedInputNode:
    """
    Seed input with control_after_generate.
    """

    RETURN_TYPES = (IO.INT,)
    RETURN_NAMES = ("seed",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/inputs/integer"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": (IO.INT, {
                    "default": 1,
                    "min": 1,
                    "max": 100000,
                    "control_after_generate": True,
                    "tooltip": "Seed for reproducibility"
                }),
            },
        }

    def execute(self, seed: int = 1):
        return (seed,)


class VisualIntensityInputNode:
    """
    Visual intensity selector.
    """

    RETURN_TYPES = (IO.INT,)
    RETURN_NAMES = ("visual_intensity",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/inputs/integer"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "visual_intensity": (IO.INT, {
                    "default": 6,
                    "min": 2,
                    "max": 10,
                    "tooltip": "Visual intensity of generation"
                }),
            },
        }

    def execute(self, visual_intensity: int = 6):
        return (visual_intensity,)


class StyleStrengthInputNode:
    """
    Style strength selector.
    """

    RETURN_TYPES = (IO.INT,)
    RETURN_NAMES = ("style_strength",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/inputs/integer"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "strength": (IO.INT, {
                    "default": 50,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Style application strength"
                }),
            },
        }

    def execute(self, strength: int = 50):
        return (strength,)


class StructureStrengthInputNode:
    """
    Structure strength selector.
    """

    RETURN_TYPES = (IO.INT,)
    RETURN_NAMES = ("structure_strength",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/inputs/integer"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "strength": (IO.INT, {
                    "default": 50,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Structure reference strength"
                }),
            },
        }

    def execute(self, strength: int = 50):
        return (strength,)


class OutputWidthInputNode:
    """
    Output width for expand operations.
    """

    RETURN_TYPES = (IO.INT,)
    RETURN_NAMES = ("output_width",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/inputs/integer"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": (IO.INT, {
                    "default": 2048,
                    "min": 1,
                    "max": 2688,
                    "tooltip": "Output width in pixels"
                }),
            },
        }

    def execute(self, width: int = 2048):
        return (width,)


class OutputHeightInputNode:
    """
    Output height for expand operations.
    """

    RETURN_TYPES = (IO.INT,)
    RETURN_NAMES = ("output_height",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/inputs/integer"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "height": (IO.INT, {
                    "default": 2048,
                    "min": 1,
                    "max": 2688,
                    "tooltip": "Output height in pixels"
                }),
            },
        }

    def execute(self, height: int = 2048):
        return (height,)


# ============================================================================
# Passthrough Output Nodes (Variable Outputs)
# ============================================================================

# Text/String Variable Outputs
class PromptOutputNode:
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/outputs/variables"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"prompt": (IO.STRING, {"forceInput": True})}}

    def execute(self, prompt: str):
        return (prompt,)


class NegativePromptOutputNode:
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("negative_prompt",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/outputs/variables"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"negative_prompt": (IO.STRING, {"forceInput": True})}}

    def execute(self, negative_prompt: str):
        return (negative_prompt,)


class PromptSuffixOutputNode:
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("prompt_suffix",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/outputs/variables"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"suffix": (IO.STRING, {"forceInput": True})}}

    def execute(self, suffix: str):
        return (suffix,)


class CustomModelIdOutputNode:
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("custom_model_id",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/outputs/variables"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"custom_model_id": (IO.STRING, {"forceInput": True})}}

    def execute(self, custom_model_id: str):
        return (custom_model_id,)


class LocaleCodeOutputNode:
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("locale_code",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/outputs/variables"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"locale_code": (IO.STRING, {"forceInput": True})}}

    def execute(self, locale_code: str):
        return (locale_code,)


class UploadIdOutputNode:
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("upload_id",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/outputs/variables"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"upload_id": (IO.STRING, {"forceInput": True})}}

    def execute(self, upload_id: str):
        return (upload_id,)


# Enum Variable Outputs
class ContentClassOutputNode:
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("content_class",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/outputs/variables"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"content_class": (IO.STRING, {"forceInput": True})}}

    def execute(self, content_class: str):
        return (content_class,)


class SizeOutputNode:
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("size",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/outputs/variables"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"size": (IO.STRING, {"forceInput": True})}}

    def execute(self, size: str):
        return (size,)


class ModelVersionOutputNode:
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("model_version",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/outputs/variables"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_version": (IO.STRING, {"forceInput": True})}}

    def execute(self, model_version: str):
        return (model_version,)


class UpsamplerTypeOutputNode:
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("upsampler_type",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/outputs/variables"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"upsampler_type": (IO.STRING, {"forceInput": True})}}

    def execute(self, upsampler_type: str):
        return (upsampler_type,)


class StylePresetOutputNode:
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("style_preset",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/outputs/variables"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"style_preset": (IO.STRING, {"forceInput": True})}}

    def execute(self, style_preset: str):
        return (style_preset,)


# Integer Variable Outputs
class NumVariationsOutputNode:
    RETURN_TYPES = (IO.INT,)
    RETURN_NAMES = ("num_variations",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/outputs/variables"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"num_variations": (IO.INT, {"forceInput": True})}}

    def execute(self, num_variations: int):
        return (num_variations,)


class SeedOutputNode:
    RETURN_TYPES = (IO.INT,)
    RETURN_NAMES = ("seed",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/outputs/variables"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"seed": (IO.INT, {"forceInput": True})}}

    def execute(self, seed: int):
        return (seed,)


class VisualIntensityOutputNode:
    RETURN_TYPES = (IO.INT,)
    RETURN_NAMES = ("visual_intensity",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/outputs/variables"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"visual_intensity": (IO.INT, {"forceInput": True})}}

    def execute(self, visual_intensity: int):
        return (visual_intensity,)


class StyleStrengthOutputNode:
    RETURN_TYPES = (IO.INT,)
    RETURN_NAMES = ("style_strength",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/outputs/variables"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"style_strength": (IO.INT, {"forceInput": True})}}

    def execute(self, style_strength: int):
        return (style_strength,)


class StructureStrengthOutputNode:
    RETURN_TYPES = (IO.INT,)
    RETURN_NAMES = ("structure_strength",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly/outputs/variables"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"structure_strength": (IO.INT, {"forceInput": True})}}

    def execute(self, structure_strength: int):
        return (structure_strength,)


# ============================================================================
# Node Mappings
# ============================================================================

NODE_CLASS_MAPPINGS = {
    # Text input nodes
    "PromptInputNode": PromptInputNode,
    "NegativePromptInputNode": NegativePromptInputNode,
    "PromptSuffixInputNode": PromptSuffixInputNode,
    "CustomModelIdInputNode": CustomModelIdInputNode,
    "PromptBiasingLocaleInputNode": PromptBiasingLocaleInputNode,
    "StylePresetInputNode": StylePresetInputNode,
    "UploadIdInputNode": UploadIdInputNode,

    # Enum input nodes
    "ContentClassInputNode": ContentClassInputNode,
    "SizeInputNode": SizeInputNode,
    "ModelVersionInputNode": ModelVersionInputNode,
    "UpsamplerTypeInputNode": UpsamplerTypeInputNode,

    # Integer input nodes
    "NumVariationsInputNode": NumVariationsInputNode,
    "SeedInputNode": SeedInputNode,
    "VisualIntensityInputNode": VisualIntensityInputNode,
    "StyleStrengthInputNode": StyleStrengthInputNode,
    "StructureStrengthInputNode": StructureStrengthInputNode,
    "OutputWidthInputNode": OutputWidthInputNode,
    "OutputHeightInputNode": OutputHeightInputNode,

    # Variable output nodes
    "PromptOutputNode": PromptOutputNode,
    "NegativePromptOutputNode": NegativePromptOutputNode,
    "PromptSuffixOutputNode": PromptSuffixOutputNode,
    "CustomModelIdOutputNode": CustomModelIdOutputNode,
    "LocaleCodeOutputNode": LocaleCodeOutputNode,
    "UploadIdOutputNode": UploadIdOutputNode,
    "ContentClassOutputNode": ContentClassOutputNode,
    "SizeOutputNode": SizeOutputNode,
    "ModelVersionOutputNode": ModelVersionOutputNode,
    "UpsamplerTypeOutputNode": UpsamplerTypeOutputNode,
    "StylePresetOutputNode": StylePresetOutputNode,
    "NumVariationsOutputNode": NumVariationsOutputNode,
    "SeedOutputNode": SeedOutputNode,
    "VisualIntensityOutputNode": VisualIntensityOutputNode,
    "StyleStrengthOutputNode": StyleStrengthOutputNode,
    "StructureStrengthOutputNode": StructureStrengthOutputNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Text input nodes
    "PromptInputNode": "Prompt",
    "NegativePromptInputNode": "Negative Prompt",
    "PromptSuffixInputNode": "Prompt Suffix",
    "CustomModelIdInputNode": "Custom Model ID",
    "PromptBiasingLocaleInputNode": "Locale Code",
    "StylePresetInputNode": "Style Preset",
    "UploadIdInputNode": "Upload ID",

    # Enum input nodes
    "ContentClassInputNode": "Content Class",
    "SizeInputNode": "Size",
    "ModelVersionInputNode": "Model Version",
    "UpsamplerTypeInputNode": "Upsampler Type",

    # Integer input nodes
    "NumVariationsInputNode": "Num Variations",
    "SeedInputNode": "Seed",
    "VisualIntensityInputNode": "Visual Intensity",
    "StyleStrengthInputNode": "Style Strength",
    "StructureStrengthInputNode": "Structure Strength",
    "OutputWidthInputNode": "Output Width",
    "OutputHeightInputNode": "Output Height",

    # Variable output nodes
    "PromptOutputNode": "Prompt Value",
    "NegativePromptOutputNode": "Negative Prompt Value",
    "PromptSuffixOutputNode": "Suffix Value",
    "CustomModelIdOutputNode": "Model ID Value",
    "LocaleCodeOutputNode": "Locale Value",
    "UploadIdOutputNode": "Upload ID Value",
    "ContentClassOutputNode": "Content Class Value",
    "SizeOutputNode": "Size Value",
    "ModelVersionOutputNode": "Model Version Value",
    "UpsamplerTypeOutputNode": "Upsampler Value",
    "StylePresetOutputNode": "Style Preset Value",
    "NumVariationsOutputNode": "Variations Value",
    "SeedOutputNode": "Seed Value",
    "VisualIntensityOutputNode": "Intensity Value",
    "StyleStrengthOutputNode": "Style Strength Value",
    "StructureStrengthOutputNode": "Structure Strength Value",
}
