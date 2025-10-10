class TestNode:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("test_output",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/firefly"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "test_input": ("STRING", {"default": "hello"}),
            },
        }

    def execute(self, test_input="hello"):
        # Whatever comes in (connection or widget), pass it through to output
        return (test_input,)


NODE_CLASS_MAPPINGS = {
    "TestNode": TestNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TestNode": "Test Node 5",
}
