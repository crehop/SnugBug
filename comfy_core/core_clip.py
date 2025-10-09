from __future__ import annotations

import node_helpers


class CLIPVisionEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_vision": ("CLIP_VISION",),
                              "image": ("IMAGE",),
                              "crop": (["center", "none"],)
                             }}
    RETURN_TYPES = ("CLIP_VISION_OUTPUT",)
    FUNCTION = "encode"

    CATEGORY = "Legacy/conditioning"

    def encode(self, clip_vision, image, crop):
        crop_image = True
        if crop != "center":
            crop_image = False
        output = clip_vision.encode_image(image, crop=crop_image)
        return (output,)


class unCLIPConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "noise_augmentation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_adm"

    CATEGORY = "Legacy/conditioning"

    def apply_adm(self, conditioning, clip_vision_output, strength, noise_augmentation):
        if strength == 0:
            return (conditioning, )

        c = node_helpers.conditioning_set_values(conditioning, {"unclip_conditioning": [{"clip_vision_output": clip_vision_output, "strength": strength, "noise_augmentation": noise_augmentation}]}, append=True)
        return (c, )


NODE_CLASS_MAPPINGS = {
    "CLIPVisionEncode": CLIPVisionEncode,
    "unCLIPConditioning": unCLIPConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPVisionEncode": "CLIP Vision Encode",
    "unCLIPConditioning": "unCLIP Conditioning",
}
