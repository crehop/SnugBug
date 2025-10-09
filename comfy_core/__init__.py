# Import all node modules from comfy_core
from . import core_conditioning
from . import core_latent
from . import core_loaders
from . import core_controlnet
from . import core_image
from . import core_sampling
from . import core_clip
from . import core_style
from . import core_gligen

# Collect all node class mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Merge all module mappings
for module in [
    core_conditioning,
    core_latent,
    core_loaders,
    core_controlnet,
    core_image,
    core_sampling,
    core_clip,
    core_style,
    core_gligen,
]:
    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
