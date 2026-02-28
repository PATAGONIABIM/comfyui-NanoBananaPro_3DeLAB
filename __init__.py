from .nodes import GeminiNanoBananaPro, LoadScribbleImage, GeminiVeo31VideoGenerator, NanoBananaPreviewVideo

NODE_CLASS_MAPPINGS = {
    "GeminiNanoBananaPro": GeminiNanoBananaPro,
    "LoadScribbleImage": LoadScribbleImage,
    "GeminiVeo31VideoGenerator": GeminiVeo31VideoGenerator,
    "NanoBananaPreviewVideo": NanoBananaPreviewVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiNanoBananaPro": "Nano Banana Pro (Gemini Direct)",
    "LoadScribbleImage": "Load Image & Scribble Editor",
    "GeminiVeo31VideoGenerator": "Veo 3.1 Video",
    "NanoBananaPreviewVideo": "Veo Video Preview"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

WEB_DIRECTORY = "./js"
