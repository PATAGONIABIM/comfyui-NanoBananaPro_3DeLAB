from .nodes import GeminiNanoBananaPro, LoadScribbleImage

NODE_CLASS_MAPPINGS = {
    "GeminiNanoBananaPro": GeminiNanoBananaPro,
    "LoadScribbleImage": LoadScribbleImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiNanoBananaPro": "Nano Banana Pro (Gemini Direct)",
    "LoadScribbleImage": "Load Image & Scribble Editor"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

WEB_DIRECTORY = "./js"
