# Nano Banana Pro (Gemini Direct)

This node integrates Google's **Gemini** and **Vertex AI (Imagen 3)** models directly into ComfyUI for high-speed, high-volume image generation and advanced editing.

## Features
- Unlocks **Gemini 3 Pro** and **Gemini 3.1 Flash (Nano Banana 2)** for image generation.
- Supports **Imagen 3** for advanced editing tasks like Inpainting, Outpainting, and Background Swap.
- Native multiline support for prompts and direct image composition via API limits.

## Parameters

### Required
- **prompt**: The text description of the image you want to generate.
- **model**: Select the AI model to use:
  - `gemini-3.1-flash-image-preview` / `gemini-3-pro-image-preview`: Fast text/image to image generation (requires Google AI Studio API Key).
  - `imagen-*`: Advanced image editing operations (requires Vertex AI JSON Key).
- **operation**: Action to perform:
  - `GENERATE`: Standard Text-to-Image or Image-to-Image generation.
  - `INPAINT_INSERTION` / `INPAINT_REMOVAL` / `OUTPAINT` / `BACKGROUND_SWAP`: Image editing (Requires Imagen + Vertex AI).

### Optional
- **api_key**: Google AI Studio API Key (for Gemini models). Also accepts a path to a txt file.
- **service_account_json**: Absolute path to a Google Cloud Vertex AI Service Account JSON key (for Imagen 3/Editing).
- **images**: Input image for Image-to-Image or Editing operations.
- **mask**: Mask image defining the region for inpainting (White = Edit, Black = Keep).
- **scribble**: Scribble or sketch image for controlled inpainting. Easily created with the *Load Image & Scribble Editor* node.
- **seed**: Specific seed to use for generation (or -1/random for variations).
- **aspect_ratio**: Ratio of the generated output (1:1, 16:9, etc.).
- **resolution**: Quality/size of the image (1K, 2K, 4K).
- **response_modalities**: Pick `IMAGE` for standard output, or `IMAGE+TEXT` if the model supports textual reasoning.
- **system_prompt**: Overall instructions given to the model for styling or context.

## Usage Guide
1. **Gemini Generation**: Provide an `api_key` and run the node in `GENERATE` mode using a Gemini model.
2. **Imagen 3 Editing**: Provide the absolute path to a Vertex AI `service_account_json`, attach `images` / `mask` / `scribble`, set the model to `imagen-3.0-capability-001`, and select an operation like `INPAINT_INSERTION`.
