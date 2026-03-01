# Gemini Veo 3.1 & 2.0 Video Generator

This node provides video generation and editing using Google's state-of-the-art **Veo 3.1** and **Veo 2.0** models.

## Features
- **Text-to-Video**: Standard video generation using `veo-3.1-generate-preview` via Google AI Studio API Key.
- **Image-to-Video (Reference Images)**: Generate videos based on up to 3 reference images using `veo-2.0-generate-exp` via Vertex AI (Requires Service Account JSON and GCS Bucket).
- **Video Editing**: Inpaint objects into videos, remove objects, or extend existing clips using `veo-2.0-generate-preview` (Requires Vertex AI setup).

## Parameters

### Required
- **prompt**: Text prompt describing the desired video or edit.
- **model**: Select the video model (`veo-3.1-generate-preview`, `veo-2.0-generate-exp`, etc.).
- **mode**: Operation to perform (`standard`, `reference images`, `inpaint_insertion`, `inpaint_removal`, `extend_video`).
- **api_key**: Google AI Studio API Key (Required for standard Veo 3.1).
- **service_account_json**: Absolute path to a Vertex AI Service Account JSON (Required for Veo 2.0 and editing operations).
- **gcs_bucket**: Name of your Google Cloud Storage bucket (Required for uploading video/masks/images during Vertex AI operations).

### Optional Inputs
- **image**: Reference image(s) for `reference images` mode (can batch up to 3 images).
- **video**: Base video for inpainting or extension.
- **mask**: Mask image defining the region for video inpainting.

## Usage Guide
1. **Standard `txt2vid` (Veo 3.1)**: Set the mode to `standard`, use `veo-3.1-generate-preview`, and supply your `api_key`.
2. **Reference Images `img2vid` (Veo 2.0)**: Use `veo-2.0-generate-exp`, set mode to `reference images`, connect up to 3 batched `image`s, and supply your `service_account_json` and `gcs_bucket`.
3. **Video Inpainting (Veo 2.0)**: Use `veo-2.0-generate-preview`, set mode to `inpaint_insertion` or `inpaint_removal`. Connect a `video` and a `mask`. Supply `service_account_json` and `gcs_bucket` for automated asset uploading.
