"""
Adobe Firefly API Nodes for ComfyUI

Provides nodes for interacting with Adobe Firefly API v3, including:
- Text to image generation
- Generative fill (inpainting)
- Generative expand (outpainting)
- Generate similar images
- Object composite generation
- Text to video generation
- Image upload to storage
- Custom models listing
"""

from __future__ import annotations
from inspect import cleandoc
from typing import Optional
from comfy.utils import ProgressBar
from comfy.comfy_types.node_typing import IO
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
    GenerateVideoRequest,
    UploadImageRequest,
    AsyncAcceptResponse,
    AsyncTaskResponse,
    AsyncVideoTaskResponse,
    UploadImageResponse,
    CustomModelsResponse,
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
from server import PromptServer

import torch
from io import BytesIO
import logging


# ============================================================================
# Helper Functions
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
    # We'll replace the headers with Adobe OAuth headers
    client = ApiClient(
        base_url="https://firefly-api.adobe.io",
        verify_ssl=True,
        comfy_api_key="adobe_oauth",  # Dummy value to bypass _check_auth()
    )

    # Store auth headers for use in requests
    client._adobe_headers = auth_headers

    # Override get_headers to include Adobe auth instead of ComfyUI auth
    original_get_headers = client.get_headers

    def get_headers_with_adobe():
        headers = original_get_headers()
        headers.update(client._adobe_headers)
        # Remove ComfyUI auth headers since we're using Adobe OAuth
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
    # Create Adobe API client
    client = await create_adobe_client()

    try:
        # Step 1: Request upload URL
        upload_request_endpoint = ApiEndpoint(
            path="/v2/storage/image",
            method=HttpMethod.POST,
            request_model=UploadImageRequest,
            response_model=UploadImageResponse,
        )

        upload_request_op = SynchronousOperation(
            endpoint=upload_request_endpoint,
            request=UploadImageRequest(
                name="image.png",
                type=FireflyImageFormat.IMAGE_PNG,
            ),
            api_base="https://firefly-api.adobe.io",
        )

        upload_response: UploadImageResponse = await upload_request_op.execute(client=client)

        # Step 2: Upload the image to the presigned URL
        image_bytes = tensor_to_bytesio(image, total_pixels=total_pixels)

        await ApiClient.upload_file(
            upload_url=upload_response.uploadUrl,
            file=image_bytes,
            content_type=FireflyImageFormat.IMAGE_PNG.value,
        )

        return upload_response.uploadId
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
# Upload Node
# ============================================================================

class FireflyUploadImageNode:
    """
    Upload an image to Firefly storage and return the upload ID.
    This is a helper node for other Firefly nodes that require image references.
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("upload_id",)
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/firefly"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE,),
            },
        }

    async def api_call(
        self,
        image: torch.Tensor,
    ):
        # Upload first image in batch
        upload_id = await upload_image_to_firefly(
            image=image[0] if len(image.shape) == 4 else image,
        )

        return (upload_id,)


# ============================================================================
# Text to Image Node
# ============================================================================

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

# Firefly style presets
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

class FireflyTextToImageNode:
    """
    Generate images from text prompts using Adobe Firefly.
    Supports photo and art content classes with optional style and structure references.
    """

    RETURN_TYPES = (IO.IMAGE, IO.STRING, IO.STRING, IO.STRING, IO.STRING, IO.STRING)
    RETURN_NAMES = ("image", "url_1", "url_2", "url_3", "url_4", "debug_log")
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/firefly"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                # Text inputs
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for image generation (max 1024 characters).",
                    },
                ),
                "negative_prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "tooltip": "Negative prompt to exclude unwanted elements.",
                    },
                ),
                "prompt_suffix": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "tooltip": "Text to append to the main prompt.",
                    },
                ),
                "custom_model_id": (
                    IO.STRING,
                    {
                        "tooltip": "Custom model identifier (for *_custom model versions).",
                    },
                ),
                "prompt_biasing_locale": (
                    IO.STRING,
                    {
                        "tooltip": "Language/locale code (e.g., 'en-US', 'ja-JP').",
                    },
                ),
                "style_preset": (
                    FIREFLY_STYLE_PRESETS,
                    {
                        "default": "none",
                        "tooltip": "Style preset to apply to generation.",
                    },
                ),
                "style_upload_id": (
                    IO.STRING,
                    {
                        "tooltip": "Style reference image upload ID or presigned URL from Firefly storage.",
                    },
                ),
                "structure_upload_id": (
                    IO.STRING,
                    {
                        "tooltip": "Structure reference image upload ID or presigned URL from Firefly storage.",
                    },
                ),

                # Enum/Dropdown inputs
                "content_class": (
                    [c.value for c in FireflyContentClass],
                    {
                        "tooltip": "Content class: 'photo' for photorealistic, 'art' for artistic style.",
                    },
                ),
                "size": (
                    FIREFLY_ASPECT_RATIOS,
                    {
                        "default": "2048x2048 (1:1)",
                        "tooltip": "Image size and aspect ratio.",
                    },
                ),
                "model_version": (
                    ["image3", "image3_custom", "image4_standard", "image4_ultra", "image4_custom"],
                    {
                        "default": "image4_standard",
                        "tooltip": "Firefly model version.",
                    },
                ),
                "upsampler_type": (
                    ["default", "low_creativity"],
                    {
                        "tooltip": "Upsampler type (only for image4_custom model).",
                    },
                ),

                # Integer inputs
                "num_variations": (
                    IO.INT,
                    {
                        "min": 1,
                        "max": 4,
                        "tooltip": "Number of image variations to generate.",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "min": 1,
                        "max": 100000,
                        "control_after_generate": True,
                        "tooltip": "Seed for reproducibility (1-100,000).",
                    },
                ),
                "visual_intensity": (
                    IO.INT,
                    {
                        "min": 2,
                        "max": 10,
                        "tooltip": "Visual intensity of the generation (2-10).",
                    },
                ),
                "style_strength": (
                    IO.INT,
                    {
                        "min": 0,
                        "max": 100,
                        "tooltip": "Style reference strength (0-100).",
                    },
                ),
                "structure_strength": (
                    IO.INT,
                    {
                        "min": 0,
                        "max": 100,
                        "tooltip": "Structure reference strength (0-100).",
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
        content_class: str = "photo",
        size: str = "2048x2048 (1:1)",
        num_variations: int = 1,
        seed: int = 1,
        visual_intensity: int = 6,
        model_version: str = "image4_standard",
        negative_prompt: str = "",
        custom_model_id: str = "",
        prompt_biasing_locale: str = "",
        style_upload_id: str = "",
        style_strength: int = 50,
        style_preset: str = "none",
        structure_upload_id: str = "",
        structure_strength: int = 50,
        upsampler_type: str = "default",
        prompt_suffix: str = "",
        unique_id: Optional[str] = None,
    ):
        # Handle empty upsampler_type (default to "default")
        if not upsampler_type or upsampler_type == "":
            upsampler_type = "default"

        # Concatenate prompt with suffix if provided
        full_prompt = (prompt + " " + prompt_suffix).strip() if prompt_suffix else prompt

        validate_string(full_prompt, strip_whitespace=False, max_length=1024)

        # Parse size string to extract width and height
        # Format: "2048x2048 (1:1)" -> width=2048, height=2048
        size_parts = size.split(" ")[0].split("x")
        width = int(size_parts[0])
        height = int(size_parts[1])

        # Create Adobe API client with model version
        client = await create_adobe_client(model_version=model_version)

        try:
            # Build style configuration if style_upload_id is provided
            style_config = None
            if style_upload_id:
                # Detect if style_upload_id is a presigned URL or an upload ID
                if style_upload_id.lower().startswith("http"):
                    # It's a presigned URL
                    style_ref = FireflyStyleImageReferenceV3(
                        source=FireflyPublicBinaryInput(url=style_upload_id)
                    )
                else:
                    # It's an upload ID
                    style_ref = FireflyStyleImageReferenceV3(
                        source=FireflyPublicBinaryInput(uploadId=style_upload_id)
                    )
                # Use style_preset if it's not "none"
                presets_list = [style_preset] if style_preset and style_preset != "none" else None
                style_config = FireflyStyles(
                    imageReference=style_ref,
                    strength=style_strength,
                    presets=presets_list,
                )

            # Build structure configuration if structure_upload_id is provided
            structure_config = None
            if structure_upload_id:
                # Detect if structure_upload_id is a presigned URL or an upload ID
                if structure_upload_id.lower().startswith("http"):
                    # It's a presigned URL
                    structure_ref = FireflyStructureImageReferenceV3(
                        source=FireflyPublicBinaryInput(url=structure_upload_id)
                    )
                else:
                    # It's an upload ID
                    structure_ref = FireflyStructureImageReferenceV3(
                        source=FireflyPublicBinaryInput(uploadId=structure_upload_id)
                    )
                structure_config = FireflyStructure(
                    imageReference=structure_ref,
                    strength=structure_strength,
                )

            # Prepare request
            request = GenerateImagesRequest(
                prompt=full_prompt,
                contentClass=FireflyContentClass(content_class),
                customModelId=custom_model_id if (custom_model_id and model_version.endswith("_custom")) else None,
                size=FireflySize(width=width, height=height),
                numVariations=num_variations,
                seeds=[seed],
                negativePrompt=negative_prompt if negative_prompt else None,
                promptBiasingLocaleCode=prompt_biasing_locale if prompt_biasing_locale else None,
                style=style_config,
                structure=structure_config,
                visualIntensity=visual_intensity if model_version != "image4_custom" else None,
                upsamplerType=FireflyUpsamplerType(upsampler_type) if model_version == "image4_custom" else None,
            )

            # Build request body dict for logging
            request_body_dict = {
                "prompt": full_prompt,
                "contentClass": content_class,
                "size": {"width": width, "height": height},
                "numVariations": num_variations,
                "seeds": [seed],
            }
            if model_version != "image4_custom":
                request_body_dict["visualIntensity"] = visual_intensity
            if custom_model_id and model_version.endswith("_custom"):
                request_body_dict["customModelId"] = custom_model_id
            if prompt_biasing_locale:
                request_body_dict["promptBiasingLocaleCode"] = prompt_biasing_locale
            if negative_prompt:
                request_body_dict["negativePrompt"] = negative_prompt
            if style_config:
                request_body_dict["style"] = {
                    "imageReference": {"source": {"uploadId": style_upload_id}} if not style_upload_id.lower().startswith("http") else {"source": {"url": style_upload_id}},
                    "strength": style_strength,
                }
                if style_preset and style_preset != "none":
                    request_body_dict["style"]["presets"] = [style_preset]
            if structure_config:
                request_body_dict["structure"] = {
                    "imageReference": {"source": {"uploadId": structure_upload_id}} if not structure_upload_id.lower().startswith("http") else {"source": {"url": structure_upload_id}},
                    "strength": structure_strength,
                }
            if model_version == "image4_custom":
                request_body_dict["upsamplerType"] = upsampler_type

            # Console logging
            console_log = ""

            console_log += "=======================================================\n"
            console_log += "POST /v3/images/generate-async\n"
            console_log += "-------------------------------------------------------\n"
            console_log += f"Request Headers:\n"
            console_log += f"  x-model-version: {model_version}\n"
            console_log += f"  Authorization: Bearer [REDACTED]\n"
            console_log += f"  Content-Type: application/json\n"
            console_log += f"\nRequest Body:\n"
            import json
            console_log += json.dumps(request_body_dict, indent=2) + "\n"

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

            # Log response
            response_body_dict = {
                "jobId": submit_response.jobId,
            }
            console_log += f"\nResponse Status: 202 Accepted\n"
            console_log += f"Response Headers:\n"
            console_log += f"  Content-Type: application/json\n"
            console_log += f"\nResponse Body:\n"
            console_log += json.dumps(response_body_dict, indent=2) + "\n"
            console_log += "=======================================================\n"

            # Poll for completion
            console_log += f"\n=======================================================\n"
            console_log += f"GET /v3/status/{submit_response.jobId}\n"
            console_log += f"-------------------------------------------------------\n"
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

            # Log polling response
            poll_response_dict = {
                "status": result.status,
                "jobId": result.jobId,
                "outputs": [{"image": {"url": "[PRESIGNED_URL]"}} for _ in (result.outputs or [])],
            }
            console_log += f"\nResponse Status: 200 OK\n"
            console_log += f"Response Headers:\n"
            console_log += f"  Content-Type: application/json\n"
            console_log += f"\nResponse Body:\n"
            console_log += json.dumps(poll_response_dict, indent=2) + "\n"
            console_log += f"\nSummary: {len(result.outputs) if result.outputs else 0} image(s) generated\n"
            console_log += "=======================================================\n"

            # Debug: Log the response
            logging.info(f"Firefly API response: {result}")
            logging.info(f"Response status: {result.status}")
            logging.info(f"Response outputs: {result.outputs}")

            # Download outputs
            if not result.outputs:
                console_log += f"\n=======================================================\n"
                console_log += f"ERROR: No outputs in response\n"
                console_log += f"  status: {result.status}\n"
                console_log += f"  errorCode: {result.errorCode}\n"
                console_log += f"  errorMessage: {result.errorMessage}\n"
                console_log += "=======================================================\n"
                raise Exception(f"No outputs returned from Firefly API. Status: {result.status}, Full response: {result}")

            console_log += f"\n=======================================================\n"
            console_log += f"DOWNLOADING OUTPUTS\n"
            console_log += f"-------------------------------------------------------\n"
            console_log += f"Downloading {len(result.outputs)} image(s)...\n"

            output_bytesio, presigned_urls = await download_firefly_outputs(
                result.outputs,
                unique_id=unique_id,
            )

            console_log += f"✓ Downloaded {len(output_bytesio)} image(s) successfully\n"
            console_log += "=======================================================\n"

            # Add presigned URLs to console log
            console_log += f"\nPresigned URLs (valid for 1 hour):\n"
            for i, url in enumerate(presigned_urls, 1):
                console_log += f"  [{i}] {url}\n"
            console_log += "=======================================================\n"

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

            # Debug logging
            logging.info(f"FireflyTextToImageNode returning image, URLs, and debug_log")
            logging.info(f"Debug log length: {len(console_log)}")
            logging.info(f"Number of URLs: {len(presigned_urls)}")

            return (output_image, url_1, url_2, url_3, url_4, console_log)
        finally:
            await client.close()


# ============================================================================
# Generative Fill Node
# ============================================================================

class FireflyGenerativeFillNode:
    """
    Fill/inpaint masked areas of an image using Adobe Firefly.
    Requires an image and a mask to specify the area to fill.
    """

    RETURN_TYPES = (IO.IMAGE, IO.STRING)
    RETURN_NAMES = ("image", "debug_log")
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/firefly"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE,),
                "mask": (IO.MASK,),
                "num_variations": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Number of variations to generate.",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100000,
                        "control_after_generate": True,
                        "tooltip": "Seed for reproducibility (1-100,000).",
                    },
                ),
            },
            "optional": {
                "prompt": (
                    IO.STRING,
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Optional prompt to guide the fill.",
                    },
                ),
                "negative_prompt": (
                    IO.STRING,
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
        image: torch.Tensor,
        mask: torch.Tensor,
        num_variations: int,
        seed: int,
        prompt: str = "",
        negative_prompt: str = "",
        unique_id: Optional[str] = None,
    ):
        # Create Adobe API client
        client = await create_adobe_client()

        try:
            # Prepare mask tensor
            mask = resize_mask_to_image(mask, image, allow_gradient=False, add_channel_dim=True)

            images = []
            total = image.shape[0]
            pbar = ProgressBar(total)

            for i in range(total):
                # Upload image and mask
                image_upload_id = await upload_image_to_firefly(
                    image=image[i],
                )

                mask_upload_id = await upload_image_to_firefly(
                    image=mask[i:i+1],
                )

                # Prepare request
                request = FillImageRequest(
                    image=FireflyInputImage(
                        source=FireflyPublicBinaryInput(uploadId=image_upload_id)
                    ),
                    mask=FireflyInputMask(
                        source=FireflyPublicBinaryInput(uploadId=mask_upload_id)
                    ),
                    prompt=prompt if prompt else None,
                    negativePrompt=negative_prompt if negative_prompt else None,
                    numVariations=num_variations,
                    seeds=[seed],
                )

                # Submit async job
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

                submit_response: AsyncAcceptResponse = await submit_op.execute(client=client)

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

                result: AsyncTaskResponse = await poll_op.execute(client=client)

                # Download outputs
                if not result.outputs:
                    raise Exception("No outputs returned from Firefly API")

                output_bytesio = await download_firefly_outputs(
                    result.outputs,
                    unique_id=unique_id,
                )

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
            console_log = "Debug logging not yet implemented for this node"
            return (images_tensor, console_log)
        finally:
            await client.close()


# ============================================================================
# Generative Expand Node
# ============================================================================

class FireflyGenerativeExpandNode:
    """
    Expand/outpaint an image to a larger size using Adobe Firefly.
    Extends the image beyond its original boundaries.
    """

    RETURN_TYPES = (IO.IMAGE, IO.STRING)
    RETURN_NAMES = ("image", "debug_log")
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/firefly"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE,),
                "output_width": (
                    IO.INT,
                    {
                        "default": 2048,
                        "min": 1,
                        "max": 2688,
                        "tooltip": "Width of expanded image in pixels.",
                    },
                ),
                "output_height": (
                    IO.INT,
                    {
                        "default": 2048,
                        "min": 1,
                        "max": 2688,
                        "tooltip": "Height of expanded image in pixels.",
                    },
                ),
                "num_variations": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Number of variations to generate.",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100000,
                        "control_after_generate": True,
                        "tooltip": "Seed for reproducibility (1-100,000).",
                    },
                ),
            },
            "optional": {
                "prompt": (
                    IO.STRING,
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Optional prompt to guide the expansion.",
                    },
                ),
                "negative_prompt": (
                    IO.STRING,
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
        image: torch.Tensor,
        output_width: int,
        output_height: int,
        num_variations: int,
        seed: int,
        prompt: str = "",
        negative_prompt: str = "",
        unique_id: Optional[str] = None,
    ):
        # Create Adobe API client
        client = await create_adobe_client()

        try:
            images = []
            total = image.shape[0]
            pbar = ProgressBar(total)

            for i in range(total):
                # Upload image
                image_upload_id = await upload_image_to_firefly(
                    image=image[i],
                )

                # Prepare request
                request = ExpandImageRequest(
                    image=FireflyInputImage(
                        source=FireflyPublicBinaryInput(uploadId=image_upload_id)
                    ),
                    size=FireflySize(width=output_width, height=output_height),
                    prompt=prompt if prompt else None,
                    negativePrompt=negative_prompt if negative_prompt else None,
                    numVariations=num_variations,
                    seeds=[seed],
                )

                # Submit async job
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

                submit_response: AsyncAcceptResponse = await submit_op.execute(client=client)

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

                result: AsyncTaskResponse = await poll_op.execute(client=client)

                # Download outputs
                if not result.outputs:
                    raise Exception("No outputs returned from Firefly API")

                output_bytesio = await download_firefly_outputs(
                    result.outputs,
                    unique_id=unique_id,
                )

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
            console_log = "Debug logging not yet implemented for this node"
            return (images_tensor, console_log)
        finally:
            await client.close()


# ============================================================================
# Generate Similar Node
# ============================================================================

class FireflyGenerateSimilarNode:
    """
    Generate similar images based on a reference image using Adobe Firefly.
    """

    RETURN_TYPES = (IO.IMAGE, IO.STRING)
    RETURN_NAMES = ("image", "debug_log")
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/firefly"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE,),
                "num_variations": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Number of variations to generate.",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100000,
                        "control_after_generate": True,
                        "tooltip": "Seed for reproducibility (1-100,000).",
                    },
                ),
            },
            "optional": {
                "prompt": (
                    IO.STRING,
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Optional prompt to guide the generation.",
                    },
                ),
                "negative_prompt": (
                    IO.STRING,
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
        image: torch.Tensor,
        num_variations: int,
        seed: int,
        prompt: str = "",
        negative_prompt: str = "",
        unique_id: Optional[str] = None,
    ):
        # Create Adobe API client
        client = await create_adobe_client()

        try:
            images = []
            total = image.shape[0]
            pbar = ProgressBar(total)

            for i in range(total):
                # Upload image
                image_upload_id = await upload_image_to_firefly(
                    image=image[i],
                )

                # Prepare request
                request = GenerateSimilarImagesRequest(
                    image=FireflyInputImage(
                        source=FireflyPublicBinaryInput(uploadId=image_upload_id)
                    ),
                    prompt=prompt if prompt else None,
                    negativePrompt=negative_prompt if negative_prompt else None,
                    numVariations=num_variations,
                    seeds=[seed],
                )

                # Submit async job
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

                submit_response: AsyncAcceptResponse = await submit_op.execute(client=client)

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

                result: AsyncTaskResponse = await poll_op.execute(client=client)

                # Download outputs
                if not result.outputs:
                    raise Exception("No outputs returned from Firefly API")

                output_bytesio = await download_firefly_outputs(
                    result.outputs,
                    unique_id=unique_id,
                )

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
            console_log = "Debug logging not yet implemented for this node"
            return (images_tensor, console_log)
        finally:
            await client.close()


# ============================================================================
# Generate Object Composite Node
# ============================================================================

class FireflyGenerateObjectCompositeNode:
    """
    Generate and composite an object into a background scene using Adobe Firefly.
    Requires a background image, a mask for object placement, and a prompt describing the object.
    """

    RETURN_TYPES = (IO.IMAGE, IO.STRING)
    RETURN_NAMES = ("image", "debug_log")
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/firefly"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE,),
                "mask": (IO.MASK,),
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt describing the object to generate and composite.",
                    },
                ),
                "num_variations": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Number of variations to generate.",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100000,
                        "control_after_generate": True,
                        "tooltip": "Seed for reproducibility (1-100,000).",
                    },
                ),
            },
            "optional": {
                "negative_prompt": (
                    IO.STRING,
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
        image: torch.Tensor,
        mask: torch.Tensor,
        prompt: str,
        num_variations: int,
        seed: int,
        negative_prompt: str = "",
        unique_id: Optional[str] = None,
    ):
        validate_string(prompt, strip_whitespace=False, max_length=1024)

        # Create Adobe API client
        client = await create_adobe_client()

        try:
            # Prepare mask tensor
            mask = resize_mask_to_image(mask, image, allow_gradient=False, add_channel_dim=True)

            images = []
            total = image.shape[0]
            pbar = ProgressBar(total)

            for i in range(total):
                # Upload image and mask
                image_upload_id = await upload_image_to_firefly(
                    image=image[i],
                )

                mask_upload_id = await upload_image_to_firefly(
                    image=mask[i:i+1],
                )

                # Prepare request
                request = GenerateObjectCompositeRequest(
                    image=FireflyInputImage(
                        source=FireflyPublicBinaryInput(uploadId=image_upload_id)
                    ),
                    mask=FireflyInputMask(
                        source=FireflyPublicBinaryInput(uploadId=mask_upload_id)
                    ),
                    prompt=prompt,
                    negativePrompt=negative_prompt if negative_prompt else None,
                    numVariations=num_variations,
                    seeds=[seed],
                )

                # Submit async job
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

                submit_response: AsyncAcceptResponse = await submit_op.execute(client=client)

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

                result: AsyncTaskResponse = await poll_op.execute(client=client)

                # Download outputs
                if not result.outputs:
                    raise Exception("No outputs returned from Firefly API")

                output_bytesio = await download_firefly_outputs(
                    result.outputs,
                    unique_id=unique_id,
                )

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
            console_log = "Debug logging not yet implemented for this node"
            return (images_tensor, console_log)
        finally:
            await client.close()


# ============================================================================
# Output Nodes
# ============================================================================

class URLOutputNode:
    """
    Display and manage Firefly presigned image URLs.
    Shows URLs with expiration warning and copy-friendly formatting.
    """

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "display"
    CATEGORY = "api node/image/firefly/outputs"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "presigned_urls": (IO.STRING, {"forceInput": True}),
            },
        }

    def display(self, presigned_urls: str):
        urls = [u.strip() for u in presigned_urls.split("\n") if u.strip()]

        output = "=" * 70 + "\n"
        output += "FIREFLY PRESIGNED IMAGE URLS\n"
        output += "=" * 70 + "\n\n"

        for i, url in enumerate(urls, 1):
            output += f"Image {i}:\n"
            output += f"{url}\n\n"

        output += f"Total: {len(urls)} URL(s)\n"
        output += "⚠️  URLs expire after 1 hour from generation\n"
        output += "=" * 70 + "\n"

        return {"ui": {"text": [output]}}


# ============================================================================
# Console Display Node
# ============================================================================

class FireflyConsoleNode:
    """
    Display debug console output from Firefly API nodes.
    Shows API requests, responses, and execution details in a terminal-style display.
    Connect multiple debug log outputs to aggregate logs from different nodes.
    """

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "display"
    CATEGORY = "api node/image/firefly"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "debug_log_1": (IO.STRING, {"forceInput": True}),
                "debug_log_2": (IO.STRING, {"forceInput": True}),
                "debug_log_3": (IO.STRING, {"forceInput": True}),
                "debug_log_4": (IO.STRING, {"forceInput": True}),
                "debug_log_5": (IO.STRING, {"forceInput": True}),
                "debug_log_6": (IO.STRING, {"forceInput": True}),
                "debug_log_7": (IO.STRING, {"forceInput": True}),
                "debug_log_8": (IO.STRING, {"forceInput": True}),
            },
        }

    def display(self, **kwargs):
        # Debug logging
        logging.info(f"FireflyConsoleNode.display called with kwargs: {list(kwargs.keys())}")

        # Combine all provided debug logs
        logs = []
        for key in sorted(kwargs.keys()):
            if key in kwargs and kwargs[key]:
                logging.info(f"Adding log from {key}, length: {len(kwargs[key])}")
                logs.append(kwargs[key])

        # Create terminal-style output
        if logs:
            combined_log = "\n\n".join(logs)
        else:
            combined_log = ""

        # Prepend initialization message
        terminal_output = "[INITIALIZED]\n\n" + combined_log if combined_log else "[INITIALIZED]\n\nNo debug logs connected"

        logging.info(f"FireflyConsoleNode returning output, length: {len(terminal_output)}")

        # Return as text - ComfyUI will display it
        return {"ui": {"text": [terminal_output]}}


# ============================================================================
# Node Mappings
# ============================================================================

# Import input nodes
from comfy_api_nodes.nodes_firefly_inputs import (
    NODE_CLASS_MAPPINGS as INPUT_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as INPUT_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {
    # Main generation nodes
    "FireflyUploadImageNode": FireflyUploadImageNode,
    "FireflyTextToImageNode": FireflyTextToImageNode,
    "FireflyGenerativeFillNode": FireflyGenerativeFillNode,
    "FireflyGenerativeExpandNode": FireflyGenerativeExpandNode,
    "FireflyGenerateSimilarNode": FireflyGenerateSimilarNode,
    "FireflyGenerateObjectCompositeNode": FireflyGenerateObjectCompositeNode,

    # Output nodes
    "URLOutputNode": URLOutputNode,
    "FireflyConsoleNode": FireflyConsoleNode,
}

# Merge input nodes into main mappings
NODE_CLASS_MAPPINGS.update(INPUT_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    # Main generation nodes
    "FireflyUploadImageNode": "Upload Image",
    "FireflyTextToImageNode": "Text to Image",
    "FireflyGenerativeFillNode": "Generative Fill",
    "FireflyGenerativeExpandNode": "Generative Expand",
    "FireflyGenerateSimilarNode": "Generate Similar",
    "FireflyGenerateObjectCompositeNode": "Object Composite",

    # Output nodes
    "URLOutputNode": "Presigned URLs",
    "FireflyConsoleNode": "Debug Console",
}

# Merge input node display names
NODE_DISPLAY_NAME_MAPPINGS.update(INPUT_NODE_DISPLAY_NAME_MAPPINGS)
