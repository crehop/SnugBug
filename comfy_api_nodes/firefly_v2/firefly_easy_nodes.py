"""
Adobe Firefly API Nodes V2 - Refactored with ComfyUI-EasyNodes

This is a cleaner, more maintainable version of the Firefly API nodes
using the EasyNodes framework for simplified node creation.
"""

from __future__ import annotations
from typing import Optional, Any
import torch
import logging

from comfy_api_nodes.apis.firefly_api import (
    FireflyContentClass,
    FireflyTaskStatus,
    FireflyPromptBiasingLocale,
    FireflyImageFormat,
    FireflySize,
    FireflyPublicBinaryInput,
    FireflyInputImage,
    FireflyInputMask,
    FireflyStyles,
    FireflyStructure,
    FireflyStyleImageReferenceV3,
    FireflyStructureImageReferenceV3,
    FireflyUpsamplerType,
    GenerateImagesRequest,
    FillImageRequest,
    ExpandImageRequest,
    GenerateSimilarImagesRequest,
    GenerateObjectCompositeRequest,
    UploadImageRequest,
    AsyncAcceptResponse,
    AsyncTaskResponse,
    UploadImageResponse,
)
from comfy_api_nodes.apis.client import (
    ApiClient,
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)
from comfy_api_nodes.apis.adobe_auth import get_adobe_auth_manager
from comfy_api_nodes.apinode_utils import (
    bytesio_to_image_tensor,
    download_url_to_bytesio,
    tensor_to_bytesio,
    resize_mask_to_image,
    validate_string,
)
from comfy.utils import ProgressBar
from io import BytesIO


# ============================================================================
# Helper Functions (Shared with original nodes)
# ============================================================================

async def create_adobe_client(model_version: str = "image3") -> ApiClient:
    """
    Create an ApiClient configured for Adobe Firefly API with OAuth authentication.

    Args:
        model_version: Firefly model version (e.g., "image3", "image4_standard")

    Returns:
        ApiClient instance with Adobe authentication
    """
    auth_manager = get_adobe_auth_manager()
    auth_headers = await auth_manager.get_auth_headers()

    # Add model version header
    auth_headers["x-model-version"] = model_version

    # Create client with dummy auth to bypass ComfyUI login check
    client = ApiClient(
        base_url="https://firefly-api.adobe.io",
        verify_ssl=True,
        comfy_api_key="adobe_oauth",
    )

    # Store auth headers for use in requests
    client._adobe_headers = auth_headers

    # Override get_headers to include Adobe auth
    original_get_headers = client.get_headers

    def get_headers_with_adobe():
        headers = original_get_headers()
        headers.update(client._adobe_headers)
        headers.pop("X-API-KEY", None)
        return headers

    client.get_headers = get_headers_with_adobe

    return client


async def upload_image_to_firefly(
    image: torch.Tensor,
    total_pixels: int = 4096 * 4096,
) -> str:
    """
    Upload an image to Firefly storage and return the upload ID.

    Args:
        image: Image tensor to upload
        total_pixels: Maximum total pixels for the image

    Returns:
        Upload ID from Firefly storage
    """
    client = await create_adobe_client()

    try:
        # Convert tensor to bytes
        image_bytes = tensor_to_bytesio(image, total_pixels=total_pixels)
        image_bytes.seek(0)  # Reset buffer position
        data = image_bytes.read()

        # Build headers
        headers = client.get_headers()
        headers["Content-Type"] = FireflyImageFormat.IMAGE_PNG.value

        # Make direct HTTP request with raw binary data
        url = client.base_url.rstrip("/") + "/v2/storage/image"
        session = await client._get_session()

        async with session.post(url, data=data, headers=headers, ssl=client.verify_ssl) as resp:
            resp.raise_for_status()
            response_json = await resp.json()

        # Extract upload ID from response {"images": [{"id": "..."}]}
        if "images" in response_json and len(response_json["images"]) > 0:
            upload_id = response_json["images"][0]["id"]
            return upload_id
        else:
            raise Exception(f"Unexpected response format: {response_json}")

    except Exception as e:
        raise Exception(f"Failed to upload image to Firefly: {str(e)}")
    finally:
        await client.close()


async def download_firefly_outputs(
    outputs: list,
    timeout: int = 1024,
    unique_id: Optional[str] = None,
) -> tuple[list[BytesIO], list[str]]:
    """
    Download output images from Firefly API response and capture presigned URLs.

    Args:
        outputs: List of output objects with image references
        timeout: Download timeout in seconds
        unique_id: Node unique ID for progress updates

    Returns:
        Tuple of (bytesio_list, url_list) containing downloaded images and their presigned URLs
    """
    all_bytesio = []
    all_urls = []

    for output in outputs:
        if hasattr(output, 'image') and output.image and output.image.url:
            url = output.image.url
            all_urls.append(url)
            all_bytesio.append(await download_url_to_bytesio(url, timeout=timeout))

    return all_bytesio, all_urls


# ============================================================================
# Constant Definitions
# ============================================================================

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

FIREFLY_STYLE_PRESETS = [
    "none",
    "art",
    "photo",
    "graphic",
    "bw",
    "color_pop",
    "warm_tone",
    "cool_tone",
    "golden_hour",
    "pastel",
    "cyberpunk",
    "steampunk",
    "anime",
    "concept_art",
    "cinematic",
    "dramatic",
    "minimalist",
    "vintage",
]


# ============================================================================
# V2 Node Implementation
# ============================================================================

# Note: While we explored EasyNodes, the async nature of Firefly API and complex
# optional parameters are better served by traditional node implementation with
# improved structure. The V2 version focuses on clean architecture and better
# organization rather than framework-specific features.

class FireflyUploadImageNodeV2:
    """
    Upload an image to Firefly storage and return the upload ID.
    This is a helper node for other Firefly nodes that require image references.

    V2 Improvements:
    - Cleaner code structure
    - Better error handling
    - Improved documentation
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("upload_id",)
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/firefly v2"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    async def api_call(self, image: torch.Tensor):
        """Upload first image in batch to Firefly storage."""
        upload_id = await upload_image_to_firefly(
            image=image[0] if len(image.shape) == 4 else image,
        )
        return (upload_id,)


class FireflyTextToImageNodeV2:
    """
    Generate images from text prompts using Adobe Firefly.

    V2 Improvements:
    - Cleaner parameter organization
    - Better default values
    - Improved documentation
    - Enhanced debug logging
    """

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url_1", "image_url_2", "image_url_3", "image_url_4", "debug_log")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/firefly v2"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                # Text inputs (at top)
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for image generation (max 1024 characters).",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Negative prompt to exclude unwanted elements.",
                    },
                ),
                "prompt_suffix": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text to append to the main prompt.",
                    },
                ),

                # Seed (comma-separated for multiple values)
                "seed": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Seed(s) for reproducibility. Can be single (e.g. '123') or comma-separated (e.g. '1,2,3,4') for multiple variations.",
                    },
                ),

                # Primary controls
                "model_version": (
                    ["image3", "image3_custom", "image4_standard", "image4_ultra", "image4_custom"],
                    {
                        "default": "image4_standard",
                        "tooltip": "Firefly model version.",
                    },
                ),
                "size": (
                    FIREFLY_ASPECT_RATIOS,
                    {
                        "default": "2048x2048 (1:1)",
                        "tooltip": "Image size and aspect ratio.",
                    },
                ),
                "num_variations": (
                    "INT",
                    {
                        "min": 1,
                        "max": 4,
                        "default": 1,
                        "tooltip": "Number of image variations to generate.",
                    },
                ),
                "content_class": (
                    ["photo", "art"],
                    {
                        "default": "photo",
                        "tooltip": "Content class: 'photo' for photorealistic, 'art' for artistic style.",
                    },
                ),
                "style_preset": (
                    FIREFLY_STYLE_PRESETS,
                    {
                        "default": "none",
                        "tooltip": "Style preset to apply to generation.",
                    },
                ),
                "upsampler_type": (
                    ["default", "low_creativity"],
                    {
                        "default": "default",
                        "tooltip": "Upsampler type (only for image4_custom model).",
                    },
                ),
                "visual_intensity": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Visual intensity of the generation (2-10). Leave empty for default.",
                    },
                ),

                # Reference inputs
                "structure_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Structure reference image upload ID or presigned URL from another node.",
                    },
                ),
                "style_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Style reference image upload ID or presigned URL from another node.",
                    },
                ),

                # Image uploads
                "structure_image": (
                    "IMAGE",
                    {
                        "tooltip": "Structure reference image (auto-uploads to Firefly storage).",
                    },
                ),
                "style_image": (
                    "IMAGE",
                    {
                        "tooltip": "Style reference image (auto-uploads to Firefly storage).",
                    },
                ),

                # Strength controls
                "structure_strength": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Structure reference strength (0-100). Leave empty for default.",
                    },
                ),
                "style_strength": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Style reference strength (0-100). Leave empty for default.",
                    },
                ),

                # Advanced options
                "custom_model_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Custom model identifier (for *_custom model versions).",
                    },
                ),
                "prompt_biasing_locale": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Language/locale code (e.g., 'en-US', 'ja-JP').",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        prompt_suffix: str = "",
        custom_model_id: str = "",
        prompt_biasing_locale: str = "",
        model_version: str = "image4_standard",
        content_class: str = "photo",
        size: str = "2048x2048 (1:1)",
        num_variations: int = 1,
        seed: str = "",
        visual_intensity: str = "",
        style_preset: str = "none",
        style_image: Optional[torch.Tensor] = None,
        style_reference: str = "",
        style_strength: str = "",
        structure_image: Optional[torch.Tensor] = None,
        structure_reference: str = "",
        structure_strength: str = "",
        upsampler_type: str = "default",
        unique_id: Optional[str] = None,
    ):
        """Generate images using Adobe Firefly API."""

        # Concatenate prompt with suffix if provided
        full_prompt = (prompt + " " + prompt_suffix).strip() if prompt_suffix else prompt

        # Validate prompt
        validate_string(full_prompt, strip_whitespace=False, max_length=1024)

        # Parse size
        size_parts = size.split(" ")[0].split("x")
        width = int(size_parts[0])
        height = int(size_parts[1])

        # Parse seeds
        seeds_list = None
        if seed and seed.strip():
            try:
                seeds_list = [int(s.strip()) for s in seed.split(",") if s.strip()]
            except ValueError:
                raise ValueError(f"Invalid seed format: '{seed}'. Use integers separated by commas (e.g., '1,2,3,4').")

        # Handle empty upsampler_type (default to "default")
        if not upsampler_type or upsampler_type == "":
            upsampler_type = "default"

        # Parse visual_intensity
        visual_intensity_int = None
        if visual_intensity and visual_intensity.strip():
            try:
                visual_intensity_int = int(visual_intensity.strip())
            except ValueError:
                raise ValueError(f"Invalid visual_intensity: '{visual_intensity}'. Must be an integer between 2-10.")

        # Validate style inputs (allow at most one)
        style_inputs_provided = sum([
            style_image is not None,
            bool(style_reference),
        ])
        if style_inputs_provided > 1:
            raise ValueError("Cannot provide multiple style inputs - choose only one: 'style_image' or 'style_reference'")

        # Validate structure inputs (allow at most one)
        structure_inputs_provided = sum([
            structure_image is not None,
            bool(structure_reference),
        ])
        if structure_inputs_provided > 1:
            raise ValueError("Cannot provide multiple structure inputs - choose only one: 'structure_image' or 'structure_reference'")

        # Create Adobe API client
        client = await create_adobe_client(model_version=model_version)

        try:
            # Build style configuration
            style_config = None
            if style_image is not None or style_reference:
                # Determine style image source
                if style_image is not None:
                    # Upload to Firefly storage and get upload ID
                    upload_id = await upload_image_to_firefly(
                        image=style_image[0] if len(style_image.shape) == 4 else style_image,
                    )
                    style_ref = FireflyStyleImageReferenceV3(
                        source=FireflyPublicBinaryInput(uploadId=upload_id)
                    )
                else:
                    # Use provided upload ID or presigned URL from node connection
                    if style_reference.lower().startswith("http"):
                        style_ref = FireflyStyleImageReferenceV3(
                            source=FireflyPublicBinaryInput(url=style_reference)
                        )
                    else:
                        style_ref = FireflyStyleImageReferenceV3(
                            source=FireflyPublicBinaryInput(uploadId=style_reference)
                        )
                presets_list = [style_preset] if style_preset and style_preset != "none" else None

                # Parse style_strength
                style_strength_int = None
                if style_strength and style_strength.strip():
                    try:
                        style_strength_int = int(style_strength.strip())
                    except ValueError:
                        raise ValueError(f"Invalid style_strength: '{style_strength}'. Must be an integer between 0-100.")

                style_config = FireflyStyles(
                    imageReference=style_ref,
                    strength=style_strength_int,
                    presets=presets_list,
                )

            # Build structure configuration
            structure_config = None
            if structure_image is not None or structure_reference:
                # Determine structure image source
                if structure_image is not None:
                    # Upload to Firefly storage and get upload ID
                    upload_id = await upload_image_to_firefly(
                        image=structure_image[0] if len(structure_image.shape) == 4 else structure_image,
                    )
                    structure_ref = FireflyStructureImageReferenceV3(
                        source=FireflyPublicBinaryInput(uploadId=upload_id)
                    )
                else:
                    # Use provided upload ID or presigned URL from node connection
                    if structure_reference.lower().startswith("http"):
                        structure_ref = FireflyStructureImageReferenceV3(
                            source=FireflyPublicBinaryInput(url=structure_reference)
                        )
                    else:
                        structure_ref = FireflyStructureImageReferenceV3(
                            source=FireflyPublicBinaryInput(uploadId=structure_reference)
                        )

                # Parse structure_strength
                structure_strength_int = None
                if structure_strength and structure_strength.strip():
                    try:
                        structure_strength_int = int(structure_strength.strip())
                    except ValueError:
                        raise ValueError(f"Invalid structure_strength: '{structure_strength}'. Must be an integer between 0-100.")

                structure_config = FireflyStructure(
                    imageReference=structure_ref,
                    strength=structure_strength_int,
                )

            # Prepare request
            request = GenerateImagesRequest(
                prompt=full_prompt,
                contentClass=FireflyContentClass(content_class),
                customModelId=custom_model_id if (custom_model_id and model_version.endswith("_custom")) else None,
                size=FireflySize(width=width, height=height),
                numVariations=num_variations,
                seeds=seeds_list,
                negativePrompt=negative_prompt if negative_prompt else None,
                promptBiasingLocaleCode=prompt_biasing_locale if prompt_biasing_locale else None,
                style=style_config,
                structure=structure_config,
                visualIntensity=visual_intensity_int if (visual_intensity_int is not None and model_version != "image4_custom") else None,
                upsamplerType=FireflyUpsamplerType(upsampler_type) if model_version == "image4_custom" else None,
            )

            # Build debug log
            console_log = self._build_debug_log(
                model_version=model_version,
                prompt=full_prompt,
                prompt_suffix=prompt_suffix,
                content_class=content_class,
                width=width,
                height=height,
                num_variations=num_variations,
                seed=seed,
                visual_intensity=visual_intensity,
                negative_prompt=negative_prompt,
                custom_model_id=custom_model_id,
                prompt_biasing_locale=prompt_biasing_locale,
                style_config=style_config,
                style_image=style_image,
                style_reference=style_reference,
                style_strength=style_strength,
                style_preset=style_preset,
                structure_config=structure_config,
                structure_image=structure_image,
                structure_reference=structure_reference,
                structure_strength=structure_strength,
                upsampler_type=upsampler_type,
            )

            # Submit async job
            submit_endpoint = ApiEndpoint(
                path="/v3/images/generate-async",
                method=HttpMethod.POST,
                request_model=GenerateImagesRequest,
                response_model=AsyncAcceptResponse,
            )

            submit_op = SynchronousOperation(
                endpoint=submit_endpoint,
                request=request,
                api_base="https://firefly-api.adobe.io",
            )

            submit_response: AsyncAcceptResponse = await submit_op.execute(client=client)

            console_log += f"\nResponse: 202 Accepted\n"
            console_log += f"  jobId: {submit_response.jobId}\n"

            # Poll for completion
            console_log += f"\n{'='*55}\n"
            console_log += f"GET /v3/status/{submit_response.jobId.split(':')[-1][:8]}...\n"
            console_log += f"{'-'*55}\n"
            console_log += f"Polling for job completion...\n"

            poll_endpoint = ApiEndpoint(
                path=f"/v3/status/{submit_response.jobId}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=AsyncTaskResponse,
            )

            poll_op = PollingOperation(
                poll_endpoint=poll_endpoint,
                request=EmptyRequest(),
                completed_statuses=["succeeded"],
                failed_statuses=["failed", "canceled"],
                status_extractor=lambda x: x.status,
                api_base="https://firefly-api.adobe.io",
                poll_interval=2.0,
                max_poll_attempts=120,
                node_id=unique_id,
            )

            result: AsyncTaskResponse = await poll_op.execute(client=client)

            console_log += f"\nResponse: 200 OK\n"
            console_log += f"  status: {result.status}\n"
            console_log += f"  jobId: {result.jobId}\n"
            console_log += f"  outputs: {len(result.outputs) if result.outputs else 0} image(s)\n"

            # Validate outputs
            if not result.outputs:
                console_log += f"\n{'='*55}\n"
                console_log += f"ERROR: No outputs in response\n"
                console_log += f"  status: {result.status}\n"
                raise Exception(f"No outputs returned from Firefly API. Status: {result.status}")

            # Download outputs
            console_log += f"\n{'='*55}\n"
            console_log += f"Downloading {len(result.outputs)} output(s)...\n"

            output_bytesio, presigned_urls = await download_firefly_outputs(
                result.outputs,
                unique_id=unique_id,
            )

            console_log += f"[OK] Downloaded {len(output_bytesio)} image(s)\n"
            console_log += f"{'='*55}\n"

            # Add presigned URLs to console log
            console_log += f"\nPresigned URLs (valid for 1 hour):\n"
            for i, url in enumerate(presigned_urls, 1):
                console_log += f"  [{i}] {url}\n"
            console_log += f"{'='*55}\n"

            # Convert to tensors
            images = []
            for bytesio in output_bytesio:
                image = bytesio_to_image_tensor(bytesio)
                if len(image.shape) < 4:
                    image = image.unsqueeze(0)
                images.append(image)

            output_image = torch.cat(images, dim=0)

            # Split URLs into individual outputs (up to 4)
            url_1 = presigned_urls[0] if len(presigned_urls) > 0 else ""
            url_2 = presigned_urls[1] if len(presigned_urls) > 1 else ""
            url_3 = presigned_urls[2] if len(presigned_urls) > 2 else ""
            url_4 = presigned_urls[3] if len(presigned_urls) > 3 else ""

            return (output_image, url_1, url_2, url_3, url_4, console_log)

        finally:
            await client.close()

    def _build_debug_log(
        self,
        model_version: str,
        prompt: str,
        prompt_suffix: str,
        content_class: str,
        width: int,
        height: int,
        num_variations: int,
        seed: str,
        visual_intensity: str,
        negative_prompt: str,
        custom_model_id: str,
        prompt_biasing_locale: str,
        style_config: Any,
        style_image: Optional[torch.Tensor],
        style_reference: str,
        style_strength: str,
        style_preset: str,
        structure_config: Any,
        structure_image: Optional[torch.Tensor],
        structure_reference: str,
        structure_strength: str,
        upsampler_type: str,
    ) -> str:
        """Build formatted debug log for console output."""
        log = "=" * 55 + "\n"
        log += "POST /v3/images/generate-async\n"
        log += "-" * 55 + "\n"
        log += f"Headers:\n"
        log += f"  x-model-version: {model_version}\n"
        log += f"\nRequest Body:\n"
        log += f"  prompt: {prompt[:50]}...\n" if len(prompt) > 50 else f"  prompt: {prompt}\n"
        if prompt_suffix:
            log += f"  (suffix: '{prompt_suffix[:30]}...')\n" if len(prompt_suffix) > 30 else f"  (suffix: '{prompt_suffix}')\n"
        log += f"  contentClass: {content_class}\n"
        log += f"  size: {width}x{height}\n"
        log += f"  numVariations: {num_variations}\n"
        if seed and seed.strip():
            log += f"  seeds: [{seed}]\n"
        if visual_intensity and visual_intensity.strip() and model_version != "image4_custom":
            log += f"  visualIntensity: {visual_intensity}\n"
        if custom_model_id:
            log += f"  customModelId: {custom_model_id}\n"
        if prompt_biasing_locale:
            log += f"  promptBiasingLocaleCode: {prompt_biasing_locale}\n"
        if negative_prompt:
            log += f"  negativePrompt: {negative_prompt[:30]}...\n" if len(negative_prompt) > 30 else f"  negativePrompt: {negative_prompt}\n"

        if style_config:
            log += f"  style:\n"
            if style_image is not None:
                log += f"    uploadId: [UPLOADED IMAGE]\n"
            else:
                log += f"    uploadId: {style_reference}\n"
            if style_strength and style_strength.strip():
                log += f"    strength: {style_strength}\n"
            if style_preset and style_preset != "none":
                log += f"    preset: {style_preset}\n"

        if structure_config:
            log += f"  structure:\n"
            if structure_image is not None:
                log += f"    uploadId: [UPLOADED IMAGE]\n"
            else:
                log += f"    uploadId: {structure_reference}\n"
            if structure_strength and structure_strength.strip():
                log += f"    strength: {structure_strength}\n"

        if model_version == "image4_custom":
            log += f"  upsamplerType: {upsampler_type}\n"

        return log


class FireflyGenerativeFillNodeV2:
    """
    Fill/inpaint masked areas of an image using Adobe Firefly.

    V2 Improvements:
    - Cleaner code structure
    - Better batch processing
    - Enhanced error handling
    - Dual input support for images and masks
    """

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url_1", "image_url_2", "image_url_3", "image_url_4", "debug_log")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/firefly v2"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Image to fill (auto-uploads to Firefly storage).",
                    },
                ),
                "image_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Image upload ID or presigned URL from another node.",
                    },
                ),
                "mask": (
                    "MASK",
                    {
                        "tooltip": "Mask for fill area (auto-uploads to Firefly storage).",
                    },
                ),
                "mask_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Mask upload ID or presigned URL from another node.",
                    },
                ),
            },
            "required": {
                "num_variations": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Number of variations to generate.",
                    },
                ),
                "seed": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Seed(s) for reproducibility. Leave empty for random, or provide single seed (e.g. '12345') or multiple comma-separated seeds (e.g. '1,2,3,4').",
                    },
                ),
            },
            "optional": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Optional prompt to guide the fill.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Negative prompt to exclude unwanted elements.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        num_variations: int,
        seed: str = "",
        image: Optional[torch.Tensor] = None,
        image_reference: str = "",
        mask: Optional[torch.Tensor] = None,
        mask_reference: str = "",
        prompt: str = "",
        negative_prompt: str = "",
        unique_id: Optional[str] = None,
    ):
        """Fill masked areas using Firefly API."""
        # Validate inputs
        if image is None and not image_reference:
            raise ValueError("Must provide either 'image' or 'image_reference'")
        if image is not None and image_reference:
            raise ValueError("Cannot provide both 'image' and 'image_reference' - choose only one")
        if mask is None and not mask_reference:
            raise ValueError("Must provide either 'mask' or 'mask_reference'")
        if mask is not None and mask_reference:
            raise ValueError("Cannot provide both 'mask' and 'mask_reference' - choose only one")

        # Parse seeds
        seeds_list = None
        if seed and seed.strip():
            try:
                seeds_list = [int(s.strip()) for s in seed.split(",") if s.strip()]
            except ValueError:
                raise ValueError(f"Invalid seed format: '{seed}'. Use integers separated by commas (e.g., '1,2,3,4').")

        client = await create_adobe_client()

        try:
            # Prepare mask if provided as tensor
            if mask is not None and image is not None:
                mask = resize_mask_to_image(mask, image, allow_gradient=False, add_channel_dim=True)

            images = []
            all_urls = []
            total = image.shape[0] if image is not None else 1
            pbar = ProgressBar(total)

            for i in range(total):
                # Determine image source
                if image is not None:
                    # Upload to Firefly storage and get upload ID
                    upload_id = await upload_image_to_firefly(image=image[i])
                    image_source = FireflyPublicBinaryInput(uploadId=upload_id)
                else:
                    # Use provided upload ID or presigned URL
                    if image_reference.lower().startswith("http"):
                        image_source = FireflyPublicBinaryInput(url=image_reference)
                    else:
                        image_source = FireflyPublicBinaryInput(uploadId=image_reference)

                # Determine mask source
                if mask is not None:
                    # Upload to Firefly storage and get upload ID
                    mask_upload_id = await upload_image_to_firefly(image=mask[i:i+1])
                    mask_source = FireflyPublicBinaryInput(uploadId=mask_upload_id)
                else:
                    # Use provided upload ID or presigned URL
                    if mask_reference.lower().startswith("http"):
                        mask_source = FireflyPublicBinaryInput(url=mask_reference)
                    else:
                        mask_source = FireflyPublicBinaryInput(uploadId=mask_reference)

                # Prepare request
                request = FillImageRequest(
                    image=FireflyInputImage(
                        source=image_source
                    ),
                    mask=FireflyInputMask(
                        source=mask_source
                    ),
                    prompt=prompt if prompt else None,
                    negativePrompt=negative_prompt if negative_prompt else None,
                    numVariations=num_variations,
                    seeds=seeds_list,
                )

                # Submit and poll
                submit_endpoint = ApiEndpoint(
                    path="/v3/images/fill-async",
                    method=HttpMethod.POST,
                    request_model=FillImageRequest,
                    response_model=AsyncAcceptResponse,
                )

                submit_op = SynchronousOperation(
                    endpoint=submit_endpoint,
                    request=request,
                    api_base="https://firefly-api.adobe.io",
                )

                submit_response = await submit_op.execute(client=client)

                # Poll for completion
                poll_endpoint = ApiEndpoint(
                    path=f"/v3/status/{submit_response.jobId}",
                    method=HttpMethod.GET,
                    request_model=EmptyRequest,
                    response_model=AsyncTaskResponse,
                )

                poll_op = PollingOperation(
                    poll_endpoint=poll_endpoint,
                    request=EmptyRequest(),
                    completed_statuses=["succeeded"],
                    failed_statuses=["failed", "canceled"],
                    status_extractor=lambda x: x.status,
                    api_base="https://firefly-api.adobe.io",
                    poll_interval=5.0,
                    max_poll_attempts=120,
                    node_id=unique_id,
                )

                result = await poll_op.execute(client=client)

                # Download outputs
                if not result.outputs:
                    raise Exception("No outputs returned from Firefly API")

                output_bytesio, presigned_urls = await download_firefly_outputs(
                    result.outputs,
                    unique_id=unique_id,
                )

                all_urls.extend(presigned_urls)

                # Convert to tensors
                batch_images = []
                for bytesio in output_bytesio:
                    img = bytesio_to_image_tensor(bytesio)
                    if len(img.shape) < 4:
                        img = img.unsqueeze(0)
                    batch_images.append(img)

                images.append(torch.cat(batch_images, dim=0))
                pbar.update(1)

            images_tensor = torch.cat(images, dim=0)
            console_log = f"Generative Fill completed: {total} image(s) processed"

            # Split URLs into individual outputs (up to 4)
            url_1 = all_urls[0] if len(all_urls) > 0 else ""
            url_2 = all_urls[1] if len(all_urls) > 1 else ""
            url_3 = all_urls[2] if len(all_urls) > 2 else ""
            url_4 = all_urls[3] if len(all_urls) > 3 else ""

            return (images_tensor, url_1, url_2, url_3, url_4, console_log)

        finally:
            await client.close()


class FireflyGenerativeExpandNodeV2:
    """
    Expand/outpaint an image to a larger size using Adobe Firefly.

    V2 Improvements:
    - Cleaner code structure
    - Better batch processing
    - Enhanced error handling
    - Dual input support for images
    """

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url_1", "image_url_2", "image_url_3", "image_url_4", "debug_log")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/firefly v2"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Image to expand (auto-uploads to Firefly storage).",
                    },
                ),
                "image_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Image upload ID or presigned URL from another node.",
                    },
                ),
            },
            "required": {
                "output_width": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 1,
                        "max": 3999,
                        "tooltip": "Width of expanded image in pixels.",
                    },
                ),
                "output_height": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 1,
                        "max": 3999,
                        "tooltip": "Height of expanded image in pixels.",
                    },
                ),
                "num_variations": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Number of variations to generate.",
                    },
                ),
                "seed": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Seed(s) for reproducibility. Leave empty for random, or provide single seed (e.g. '12345') or multiple comma-separated seeds (e.g. '1,2,3,4').",
                    },
                ),
            },
            "optional": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Optional prompt to guide the expansion.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Negative prompt to exclude unwanted elements.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        output_width: int,
        output_height: int,
        num_variations: int,
        seed: str = "",
        image: Optional[torch.Tensor] = None,
        image_reference: str = "",
        prompt: str = "",
        negative_prompt: str = "",
        unique_id: Optional[str] = None,
    ):
        """Expand image using Firefly API."""
        # Validate inputs
        if image is None and not image_reference:
            raise ValueError("Must provide either 'image' or 'image_reference'")
        if image is not None and image_reference:
            raise ValueError("Cannot provide both 'image' and 'image_reference' - choose only one")

        # Parse seeds
        seeds_list = None
        if seed and seed.strip():
            try:
                seeds_list = [int(s.strip()) for s in seed.split(",") if s.strip()]
            except ValueError:
                raise ValueError(f"Invalid seed format: '{seed}'. Use integers separated by commas (e.g., '1,2,3,4').")

        client = await create_adobe_client()

        try:
            images = []
            all_urls = []
            total = image.shape[0] if image is not None else 1
            pbar = ProgressBar(total)

            for i in range(total):
                # Determine image source
                if image is not None:
                    # Upload to Firefly storage and get upload ID
                    upload_id = await upload_image_to_firefly(image=image[i])
                    image_source = FireflyPublicBinaryInput(uploadId=upload_id)
                else:
                    # Use provided upload ID or presigned URL
                    if image_reference.lower().startswith("http"):
                        image_source = FireflyPublicBinaryInput(url=image_reference)
                    else:
                        image_source = FireflyPublicBinaryInput(uploadId=image_reference)

                # Prepare request
                request = ExpandImageRequest(
                    image=FireflyInputImage(
                        source=image_source
                    ),
                    size=FireflySize(width=output_width, height=output_height),
                    prompt=prompt if prompt else None,
                    negativePrompt=negative_prompt if negative_prompt else None,
                    numVariations=num_variations,
                    seeds=seeds_list,
                )

                # Submit and poll
                submit_endpoint = ApiEndpoint(
                    path="/v3/images/expand-async",
                    method=HttpMethod.POST,
                    request_model=ExpandImageRequest,
                    response_model=AsyncAcceptResponse,
                )

                submit_op = SynchronousOperation(
                    endpoint=submit_endpoint,
                    request=request,
                    api_base="https://firefly-api.adobe.io",
                )

                submit_response = await submit_op.execute(client=client)

                # Poll for completion
                poll_endpoint = ApiEndpoint(
                    path=f"/v3/status/{submit_response.jobId}",
                    method=HttpMethod.GET,
                    request_model=EmptyRequest,
                    response_model=AsyncTaskResponse,
                )

                poll_op = PollingOperation(
                    poll_endpoint=poll_endpoint,
                    request=EmptyRequest(),
                    completed_statuses=["succeeded"],
                    failed_statuses=["failed", "canceled"],
                    status_extractor=lambda x: x.status,
                    api_base="https://firefly-api.adobe.io",
                    poll_interval=5.0,
                    max_poll_attempts=120,
                    node_id=unique_id,
                )

                result = await poll_op.execute(client=client)

                # Download outputs
                if not result.outputs:
                    raise Exception("No outputs returned from Firefly API")

                output_bytesio, presigned_urls = await download_firefly_outputs(
                    result.outputs,
                    unique_id=unique_id,
                )

                all_urls.extend(presigned_urls)

                # Convert to tensors
                batch_images = []
                for bytesio in output_bytesio:
                    img = bytesio_to_image_tensor(bytesio)
                    if len(img.shape) < 4:
                        img = img.unsqueeze(0)
                    batch_images.append(img)

                images.append(torch.cat(batch_images, dim=0))
                pbar.update(1)

            images_tensor = torch.cat(images, dim=0)
            console_log = f"Generative Expand completed: {total} image(s) processed"

            # Split URLs into individual outputs (up to 4)
            url_1 = all_urls[0] if len(all_urls) > 0 else ""
            url_2 = all_urls[1] if len(all_urls) > 1 else ""
            url_3 = all_urls[2] if len(all_urls) > 2 else ""
            url_4 = all_urls[3] if len(all_urls) > 3 else ""

            return (images_tensor, url_1, url_2, url_3, url_4, console_log)

        finally:
            await client.close()


class FireflyGenerateSimilarNodeV2:
    """
    Generate similar images based on a reference image using Adobe Firefly.

    V2 Improvements:
    - Cleaner code structure
    - Better batch processing
    - Enhanced error handling
    - Dual input support for images
    """

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url_1", "image_url_2", "image_url_3", "image_url_4", "debug_log")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/firefly v2"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Reference image (auto-uploads to Firefly storage).",
                    },
                ),
                "image_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Image upload ID or presigned URL from another node.",
                    },
                ),
            },
            "required": {
                "num_variations": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Number of variations to generate.",
                    },
                ),
                "seed": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Seed(s) for reproducibility. Leave empty for random, or provide single seed (e.g. '12345') or multiple comma-separated seeds (e.g. '1,2,3,4').",
                    },
                ),
            },
            "optional": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Optional prompt to guide the generation.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Negative prompt to exclude unwanted elements.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        num_variations: int,
        seed: str = "",
        image: Optional[torch.Tensor] = None,
        image_reference: str = "",
        prompt: str = "",
        negative_prompt: str = "",
        unique_id: Optional[str] = None,
    ):
        """Generate similar images using Firefly API."""
        # Validate inputs
        if image is None and not image_reference:
            raise ValueError("Must provide either 'image' or 'image_reference'")
        if image is not None and image_reference:
            raise ValueError("Cannot provide both 'image' and 'image_reference' - choose only one")

        # Parse seeds
        seeds_list = None
        if seed and seed.strip():
            try:
                seeds_list = [int(s.strip()) for s in seed.split(",") if s.strip()]
            except ValueError:
                raise ValueError(f"Invalid seed format: '{seed}'. Use integers separated by commas (e.g., '1,2,3,4').")

        client = await create_adobe_client()

        try:
            images = []
            all_urls = []
            total = image.shape[0] if image is not None else 1
            pbar = ProgressBar(total)

            for i in range(total):
                # Determine image source
                if image is not None:
                    # Upload to Firefly storage and get upload ID
                    upload_id = await upload_image_to_firefly(image=image[i])
                    image_source = FireflyPublicBinaryInput(uploadId=upload_id)
                else:
                    # Use provided upload ID or presigned URL
                    if image_reference.lower().startswith("http"):
                        image_source = FireflyPublicBinaryInput(url=image_reference)
                    else:
                        image_source = FireflyPublicBinaryInput(uploadId=image_reference)

                # Prepare request
                request = GenerateSimilarImagesRequest(
                    image=FireflyInputImage(
                        source=image_source
                    ),
                    prompt=prompt if prompt else None,
                    negativePrompt=negative_prompt if negative_prompt else None,
                    numVariations=num_variations,
                    seeds=seeds_list,
                )

                # Submit and poll
                submit_endpoint = ApiEndpoint(
                    path="/v3/images/generate-similar-async",
                    method=HttpMethod.POST,
                    request_model=GenerateSimilarImagesRequest,
                    response_model=AsyncAcceptResponse,
                )

                submit_op = SynchronousOperation(
                    endpoint=submit_endpoint,
                    request=request,
                    api_base="https://firefly-api.adobe.io",
                )

                submit_response = await submit_op.execute(client=client)

                # Poll for completion
                poll_endpoint = ApiEndpoint(
                    path=f"/v3/status/{submit_response.jobId}",
                    method=HttpMethod.GET,
                    request_model=EmptyRequest,
                    response_model=AsyncTaskResponse,
                )

                poll_op = PollingOperation(
                    poll_endpoint=poll_endpoint,
                    request=EmptyRequest(),
                    completed_statuses=["succeeded"],
                    failed_statuses=["failed", "canceled"],
                    status_extractor=lambda x: x.status,
                    api_base="https://firefly-api.adobe.io",
                    poll_interval=5.0,
                    max_poll_attempts=120,
                    node_id=unique_id,
                )

                result = await poll_op.execute(client=client)

                # Download outputs
                if not result.outputs:
                    raise Exception("No outputs returned from Firefly API")

                output_bytesio, presigned_urls = await download_firefly_outputs(
                    result.outputs,
                    unique_id=unique_id,
                )

                all_urls.extend(presigned_urls)

                # Convert to tensors
                batch_images = []
                for bytesio in output_bytesio:
                    img = bytesio_to_image_tensor(bytesio)
                    if len(img.shape) < 4:
                        img = img.unsqueeze(0)
                    batch_images.append(img)

                images.append(torch.cat(batch_images, dim=0))
                pbar.update(1)

            images_tensor = torch.cat(images, dim=0)
            console_log = f"Generate Similar completed: {total} image(s) processed"

            # Split URLs into individual outputs (up to 4)
            url_1 = all_urls[0] if len(all_urls) > 0 else ""
            url_2 = all_urls[1] if len(all_urls) > 1 else ""
            url_3 = all_urls[2] if len(all_urls) > 2 else ""
            url_4 = all_urls[3] if len(all_urls) > 3 else ""

            return (images_tensor, url_1, url_2, url_3, url_4, console_log)

        finally:
            await client.close()


class FireflyGenerateObjectCompositeNodeV2:
    """
    Generate and composite an object into a background scene using Adobe Firefly.

    V2 Improvements:
    - Cleaner code structure
    - Better batch processing
    - Enhanced error handling
    - Dual input support for images and masks
    """

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url_1", "image_url_2", "image_url_3", "image_url_4", "debug_log")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/firefly v2"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Background image (auto-uploads to Firefly storage).",
                    },
                ),
                "image_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Image upload ID or presigned URL from another node.",
                    },
                ),
                "mask": (
                    "MASK",
                    {
                        "tooltip": "Mask for object placement (auto-uploads to Firefly storage).",
                    },
                ),
                "mask_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Mask upload ID or presigned URL from another node.",
                    },
                ),
            },
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt describing the object to generate.",
                    },
                ),
                "num_variations": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Number of variations to generate.",
                    },
                ),
                "seed": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Seed(s) for reproducibility. Leave empty for random, or provide single seed (e.g. '12345') or multiple comma-separated seeds (e.g. '1,2,3,4').",
                    },
                ),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Negative prompt to exclude unwanted elements.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        prompt: str,
        num_variations: int,
        seed: str = "",
        image: Optional[torch.Tensor] = None,
        image_reference: str = "",
        mask: Optional[torch.Tensor] = None,
        mask_reference: str = "",
        negative_prompt: str = "",
        unique_id: Optional[str] = None,
    ):
        """Generate object composite using Firefly API."""
        validate_string(prompt, strip_whitespace=False, max_length=1024)

        # Validate inputs
        if image is None and not image_reference:
            raise ValueError("Must provide either 'image' or 'image_reference'")
        if image is not None and image_reference:
            raise ValueError("Cannot provide both 'image' and 'image_reference' - choose only one")
        if mask is None and not mask_reference:
            raise ValueError("Must provide either 'mask' or 'mask_reference'")
        if mask is not None and mask_reference:
            raise ValueError("Cannot provide both 'mask' and 'mask_reference' - choose only one")

        # Parse seeds
        seeds_list = None
        if seed and seed.strip():
            try:
                seeds_list = [int(s.strip()) for s in seed.split(",") if s.strip()]
            except ValueError:
                raise ValueError(f"Invalid seed format: '{seed}'. Use integers separated by commas (e.g., '1,2,3,4').")

        client = await create_adobe_client()

        try:
            # Prepare mask if provided as tensor
            if mask is not None and image is not None:
                mask = resize_mask_to_image(mask, image, allow_gradient=False, add_channel_dim=True)

            images = []
            all_urls = []
            total = image.shape[0] if image is not None else 1
            pbar = ProgressBar(total)

            for i in range(total):
                # Determine image source
                if image is not None:
                    # Upload to Firefly storage and get upload ID
                    upload_id = await upload_image_to_firefly(image=image[i])
                    image_source = FireflyPublicBinaryInput(uploadId=upload_id)
                else:
                    # Use provided upload ID or presigned URL
                    if image_reference.lower().startswith("http"):
                        image_source = FireflyPublicBinaryInput(url=image_reference)
                    else:
                        image_source = FireflyPublicBinaryInput(uploadId=image_reference)

                # Determine mask source
                if mask is not None:
                    # Upload to Firefly storage and get upload ID
                    mask_upload_id = await upload_image_to_firefly(image=mask[i:i+1])
                    mask_source = FireflyPublicBinaryInput(uploadId=mask_upload_id)
                else:
                    # Use provided upload ID or presigned URL
                    if mask_reference.lower().startswith("http"):
                        mask_source = FireflyPublicBinaryInput(url=mask_reference)
                    else:
                        mask_source = FireflyPublicBinaryInput(uploadId=mask_reference)

                # Prepare request
                request = GenerateObjectCompositeRequest(
                    image=FireflyInputImage(
                        source=image_source
                    ),
                    mask=FireflyInputMask(
                        source=mask_source
                    ),
                    prompt=prompt,
                    negativePrompt=negative_prompt if negative_prompt else None,
                    numVariations=num_variations,
                    seeds=seeds_list,
                )

                # Submit and poll
                submit_endpoint = ApiEndpoint(
                    path="/v3/images/generate-object-composite-async",
                    method=HttpMethod.POST,
                    request_model=GenerateObjectCompositeRequest,
                    response_model=AsyncAcceptResponse,
                )

                submit_op = SynchronousOperation(
                    endpoint=submit_endpoint,
                    request=request,
                    api_base="https://firefly-api.adobe.io",
                )

                submit_response = await submit_op.execute(client=client)

                # Poll for completion
                poll_endpoint = ApiEndpoint(
                    path=f"/v3/status/{submit_response.jobId}",
                    method=HttpMethod.GET,
                    request_model=EmptyRequest,
                    response_model=AsyncTaskResponse,
                )

                poll_op = PollingOperation(
                    poll_endpoint=poll_endpoint,
                    request=EmptyRequest(),
                    completed_statuses=["succeeded"],
                    failed_statuses=["failed", "canceled"],
                    status_extractor=lambda x: x.status,
                    api_base="https://firefly-api.adobe.io",
                    poll_interval=5.0,
                    max_poll_attempts=120,
                    node_id=unique_id,
                )

                result = await poll_op.execute(client=client)

                # Download outputs
                if not result.outputs:
                    raise Exception("No outputs returned from Firefly API")

                output_bytesio, presigned_urls = await download_firefly_outputs(
                    result.outputs,
                    unique_id=unique_id,
                )

                all_urls.extend(presigned_urls)

                # Convert to tensors
                batch_images = []
                for bytesio in output_bytesio:
                    img = bytesio_to_image_tensor(bytesio)
                    if len(img.shape) < 4:
                        img = img.unsqueeze(0)
                    batch_images.append(img)

                images.append(torch.cat(batch_images, dim=0))
                pbar.update(1)

            images_tensor = torch.cat(images, dim=0)
            console_log = f"Object Composite completed: {total} image(s) processed"

            # Split URLs into individual outputs (up to 4)
            url_1 = all_urls[0] if len(all_urls) > 0 else ""
            url_2 = all_urls[1] if len(all_urls) > 1 else ""
            url_3 = all_urls[2] if len(all_urls) > 2 else ""
            url_4 = all_urls[3] if len(all_urls) > 3 else ""

            return (images_tensor, url_1, url_2, url_3, url_4, console_log)

        finally:
            await client.close()
