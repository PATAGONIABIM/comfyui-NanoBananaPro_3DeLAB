from .nodes import GeminiNanoBananaPro, LoadScribbleImage, GeminiVeo31VideoGenerator, NanoBananaPreviewVideo, LoadVideoExtract, ImagePassthrough

NODE_CLASS_MAPPINGS = {
    "GeminiNanoBananaPro": GeminiNanoBananaPro,
    "LoadScribbleImage": LoadScribbleImage,
    "GeminiVeo31VideoGenerator": GeminiVeo31VideoGenerator,
    "NanoBananaPreviewVideo": NanoBananaPreviewVideo,
    "LoadVideoExtract_3DeLAB": LoadVideoExtract,
    "ImagePassthrough_3DeLAB": ImagePassthrough
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiNanoBananaPro": "Nano Banana Pro (Gemini Direct)",
    "LoadScribbleImage": "Load Image & Scribble Editor",
    "GeminiVeo31VideoGenerator": "Veo 3.1 Video",
    "NanoBananaPreviewVideo": "Veo Video Preview",
    "LoadVideoExtract_3DeLAB": "Load Video (Extract Frame)",
    "ImagePassthrough_3DeLAB": "Load Image (Passthrough)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

WEB_DIRECTORY = "./js"
