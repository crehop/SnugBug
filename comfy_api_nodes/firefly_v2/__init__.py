"""
Firefly V2 Nodes Package

Refactored Adobe Firefly API nodes with improved structure and organization.
This version maintains backward compatibility while providing:
- Cleaner code organization
- Better documentation
- Enhanced error handling
- Improved debug logging
"""

from .firefly_easy_nodes import (
    FireflyUploadImageNodeV2,
    FireflyTextToImageNodeV2,
    FireflyGenerativeFillNodeV2,
    FireflyGenerativeExpandNodeV2,
    FireflyGenerateSimilarNodeV2,
    FireflyGenerateObjectCompositeNodeV2,
)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FireflyUploadImageNodeV2": FireflyUploadImageNodeV2,
    "FireflyTextToImageNodeV2": FireflyTextToImageNodeV2,
    "FireflyGenerativeFillNodeV2": FireflyGenerativeFillNodeV2,
    "FireflyGenerativeExpandNodeV2": FireflyGenerativeExpandNodeV2,
    "FireflyGenerateSimilarNodeV2": FireflyGenerateSimilarNodeV2,
    "FireflyGenerateObjectCompositeNodeV2": FireflyGenerateObjectCompositeNodeV2,
}

# Display name mappings for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "FireflyUploadImageNodeV2": "Upload Image V2",
    "FireflyTextToImageNodeV2": "Text to Image V2",
    "FireflyGenerativeFillNodeV2": "Generative Fill V2",
    "FireflyGenerativeExpandNodeV2": "Generative Expand V2",
    "FireflyGenerateSimilarNodeV2": "Generate Similar V2",
    "FireflyGenerateObjectCompositeNodeV2": "Object Composite V2",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "FireflyUploadImageNodeV2",
    "FireflyTextToImageNodeV2",
    "FireflyGenerativeFillNodeV2",
    "FireflyGenerativeExpandNodeV2",
    "FireflyGenerateSimilarNodeV2",
    "FireflyGenerateObjectCompositeNodeV2",
]
