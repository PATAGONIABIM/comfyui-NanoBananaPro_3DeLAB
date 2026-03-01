import os
import torch
import numpy as np
from PIL import Image, ImageOps
import io
import base64
import requests
import json
import folder_paths
import random
import sys
import hashlib
import time

# DEBUG: Ensure User Site Packages is included (ComfyUI sometimes misses this)
user_site_packages = r"C:\Users\chris\AppData\Roaming\Python\Python312\site-packages"
if user_site_packages not in sys.path:
    sys.path.append(user_site_packages)

print(f"[NanoBananaPro] Python Executable: {sys.executable}")
print(f"[NanoBananaPro] Sys Path included: {user_site_packages in sys.path}")

try:
    from google import genai
    from google.genai.types import (
        RawReferenceImage,
        MaskReferenceImage,
        MaskReferenceConfig,
        ControlReferenceImage,
        ControlReferenceConfig,
        ControlReferenceType,
        EditImageConfig,
        GenerateImagesConfig,
        GenerateContentConfig,
        Modality
    )
    from google.genai import types, errors
    GOOGLE_GENAI_AVAILABLE = True
    print(f"[NanoBananaPro] google-genai successfully imported.")
except ImportError as e:
    GOOGLE_GENAI_AVAILABLE = False
except ImportError as e:
    GOOGLE_GENAI_AVAILABLE = False
    print(f"[NanoBananaPro] Failed to import google-genai: {e}")

class LoadScribbleImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                "hidden": {"scribble_data": "STRING"},
                }

    CATEGORY = "3DELAB"
    COLOR = "#940000"

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("Image", "scribble_mask")
    FUNCTION = "load_image"

    @classmethod
    def IS_CHANGED(s, image, scribble_data=None):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        # Also include scribble data in the hash if it exists so it updates when scribbling
        if scribble_data:
            m.update(scribble_data.encode("utf-8"))
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image, scribble_data=None):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

    def load_image(self, image, scribble_data=None):
        image_path = folder_paths.get_annotated_filepath(image)
        try:
            # Load Base Image
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image_rgb = i.convert("RGB")
            image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_rgb)[None,]
        except Exception as e:
            print(f"[NanoBananaPro] Error loading image {image_path}: {e}")
            image_tensor = torch.zeros((1, 64, 64, 3))

        # Load Scribble Data (Base64 PNG) -> RGBA Tensor
        if scribble_data and scribble_data.startswith("data:image/png;base64,"):
            try:
                b64_str = scribble_data.split(",")[1]
                img_data = base64.b64decode(b64_str)
                scribble_pil = Image.open(io.BytesIO(img_data)).convert("RGBA")
                scribble_np = np.array(scribble_pil).astype(np.float32) / 255.0
                scribble_tensor = torch.from_numpy(scribble_np)[None,]
            except Exception as e:
                print(f"[NanoBananaPro] Error decoding scribble_data: {e}")
                scribble_tensor = torch.zeros((1, 64, 64, 4)) # RGBA fallback
        else:
            scribble_tensor = torch.zeros((1, 64, 64, 4)) # Empty RGBA

        return (image_tensor, scribble_tensor)

class GeminiNanoBananaPro:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe the image you want to generate...",
                    "tooltip": "The text description of the image you want to generate."
                }),
                "model": (["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview", "imagen-3.0-capability-001", "imagen-3.0-generate-001", "imagen-4.0-generate-001"], {
                    "default": "gemini-3-pro-image-preview",
                    "tooltip": "Select the AI model to use (API Key for Gemini models to generate | JSON Key for Imagen variants for Inpaint, Outpaint, Background Swap)."
                }),
                "operation": (["GENERATE", "INPAINT_INSERTION", "INPAINT_REMOVAL", "OUTPAINT", "BACKGROUND_SWAP"], {
                    "default": "GENERATE",
                    "tooltip": "Choose the operation mode: GEN (TXT2IMG|IMG2IMG|MULTIREFERENCE), INPAINT (Edit|Remove), OUTPAINT, BG_SWAP."
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Gemini API Key (AI Studio) - For Gemini 3 Pro model",
                    "tooltip": "Your Google AI Studio API Key. Required for Gemini models."
                }),
                "service_account_json": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Path to JSON Key (Vertex AI) - For Imagen variants models",
                    "tooltip": "Absolute path to your Vertex AI Service Account JSON key file. Required for Imagen 3 editing/inpainting."
                }),
                "images": ("IMAGE", {"tooltip": "Input image for editing, inpainting, or image-to-image generation."}),
                "mask": ("MASK", {"tooltip": "Mask image for inpainting (white = edit, black = keep)."}),
                "scribble": ("IMAGE", {"tooltip": "Scribble or sketch image (transparent BG) for controlled editing."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for random number generation."}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2"], {
                    "default": "1:1",
                    "tooltip": "The aspect ratio of the generated image."
                }),
                "resolution": (["1K", "2K", "4K"], {
                    "default": "1K",
                    "tooltip": "Resolution of the output image (1K, 2K, 4K)."
                }),
                "response_modalities": (["IMAGE", "IMAGE+TEXT"], {
                    "default": "IMAGE",
                    "tooltip": "Choose image only or image + reasoning text."
                }),
                "thinking": ("BOOLEAN", {
                    "default": False,
                    "label_on": "High",
                    "label_off": "Minimal",
                    "tooltip": "Enable High thinking level (gemini-3.1 flash/pro only)."
                }),
                # "files": ("STRING", {
                #     "multiline": True, 
                #     "placeholder": "Path to local file (PDF/TXT)",
                #     "tooltip": "Path to local text or PDF files to use as context."
                # }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are an expert image-generation engine. You must ALWAYS produce an image.",
                    "tooltip": "System-level instructions to guide the model's behavior."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "response_text")
    FUNCTION = "generate"
    CATEGORY = "3DELAB"
    COLOR = "#940000"

    def generate(self, prompt, model, operation, api_key="", service_account_json="", seed=None, aspect_ratio="1:1", resolution="1K", response_modalities="IMAGE", thinking=False, images=None, mask=None, scribble=None, files=None, system_prompt=""):
        scribble_mask = scribble
        # Defaults for hidden inputs
        project_id = ""
        location = "us-central1"

        print(f"--- [NanoBananaPro] Starting {operation} ---")
        print(f"Model: {model}, Aspect Ratio: {aspect_ratio}")
        
        # Clean inputs
        api_key = api_key.strip()
        service_account_json = service_account_json.strip()
        project_id = project_id.strip()

        # --- FIX: Vertex AI Project ID/Location Correction ---
        print(f"[NanoBananaPro] Input Project ID: '{project_id}', Location: '{location}'")

        # 1. Fix commonly mistaken location
        if location == "us-central":
            print(f"[NanoBananaPro] Correcting location 'us-central' to 'us-central1'")
            location = "us-central1"
        
        # 1.5Force Global for Gemini models (Experimental)
        if model in ["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"]:
            print(f"[NanoBananaPro] Forcing location 'global' for {model} (Experimental)")
            location = "global"

        # 2. Fix ComfyUI passing Operation as Project ID (known issue)
        if project_id == operation or project_id in ["INPAINT_REMOVAL", "INPAINT_INSERTION", "OUTPAINT", "BACKGROUND_SWAP", "GENERATE"]: 
             print(f"[NanoBananaPro] Warning: Project ID matches operation name '{project_id}'. Ignoring it to use auto-detection.")
             project_id = ""

        # --- AUTHENTICATION RESOLUTION ---
        client = None
        using_vertex = False
        final_api_key = api_key # Fallback for Gemini REST path

        # Read API key if it's a file path
        if final_api_key and os.path.isfile(final_api_key):
             try:
                with open(final_api_key, 'r') as f:
                    final_api_key = f.read().strip()
                print("Loaded API Key from file.")
             except: pass

        # 1. Force Google AI Studio for Gemini models payload formatting/endpoints
        if model in ["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"]:
             if not final_api_key:
                  return (torch.zeros((1, 64, 64, 3)), f"Error: '{model}' requires an API Key (Google AI Studio).")
             
             print(f"[NanoBananaPro] Using Google AI Studio API Key for {model}")
             if "GOOGLE_GENAI_USE_VERTEXAI" in os.environ:
                 del os.environ["GOOGLE_GENAI_USE_VERTEXAI"]
                 
             try:
                 client = genai.Client(api_key=final_api_key)
             except Exception as e:
                 return (torch.zeros((1, 64, 64, 3)), f"Error Initializing Gemini Client: {e}")

        # 2. Try Vertex AI (Service Account) for Imagen variants
        elif service_account_json:
            if not os.path.exists(service_account_json):
                return (torch.zeros((1, 64, 64, 3)), f"Error: JSON Key file not found at: {service_account_json}")
            
            if not GOOGLE_GENAI_AVAILABLE:
                 return (torch.zeros((1, 64, 64, 3)), "Error: 'google-genai' library missing. Install it for Vertex AI.")

            print(f"[NanoBananaPro] Using Service Account JSON: {service_account_json}")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_json
            using_vertex = True
            
            # Auto-detect Project ID if missing
            if not project_id:
                try:
                    with open(service_account_json, 'r') as f:
                        creds = json.load(f)
                        if "project_id" in creds:
                            project_id = creds["project_id"]
                            print(f"[NanoBananaPro] Auto-detected Project ID: {project_id}")
                except Exception as e:
                    print(f"[NanoBananaPro] Warning: Could not parse Project ID from JSON: {e}")
            
            if not project_id:
                 return (torch.zeros((1, 64, 64, 3)), "Error: Project ID could not be determined. Please enter it manually.")

            try:
                os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
                client = genai.Client(vertexai=True, project=project_id, location=location)
            except Exception as e:
                 return (torch.zeros((1, 64, 64, 3)), f"Error Initializing Vertex Client: {e}")
        
        else:
             return (torch.zeros((1, 64, 64, 3)), "Error: No Authentication provided. Please enter 'api_key' (Gemini) or 'service_account_json' (Vertex).")
             
        # Check for Vertex Requirement
        is_editing = operation != "GENERATE"
        is_json_key_path = using_vertex

        # Helper to convert Tensor to PIL
        def tensor_to_pil(tensor):
            i_np = (tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(i_np)
            return img

        # Helper to convert Tensor to B64
        def tensor_to_b64(tensor): # [H, W, C]
            img = tensor_to_pil(tensor)
            buffered = io.BytesIO()
            if img.mode == "RGBA":
                img.save(buffered, format="PNG")
                mime_type = "image/png"
            else:
                img.save(buffered, format="JPEG", quality=85)
                mime_type = "image/jpeg"
            return base64.b64encode(buffered.getvalue()).decode("utf-8"), mime_type
        
        # Helper to convert Tensor to Bytes
        def tensor_to_bytes(tensor): # [H, W, C]
            img = tensor_to_pil(tensor)
            buffered = io.BytesIO()
            if img.mode == "RGBA":
                img.save(buffered, format="PNG")
                mime_type = "image/png"
            else:
                img.save(buffered, format="JPEG", quality=85)
                mime_type = "image/jpeg"
            return buffered.getvalue(), mime_type

        # --- VERTEX AI CLIENT INIT (Already done above) ---

        # --- OPERATION ROUTING ---
        
        # 1. EDITING (Inpaint/Outpaint) -> Requires Vertex/Imagen
        if is_editing:
            if not client:
                 msg = "Error: Editing (Imagen 3) requires a Service Account JSON Key (Vertex AI). Please provide the .json file path in 'api_key'."
                 print(f"[NanoBananaPro] {msg}")
                 return (torch.zeros((1, 64, 64, 3)), msg)
            
            try:
                # Prepare Inputs
                if images is None:
                     return (torch.zeros((1, 64, 64, 3)), "Error: Input image required for editing.")
                
                 # Prepare Inputs
                if images is None:
                     return (torch.zeros((1, 64, 64, 3)), "Error: Input image required for editing.")
                
                # --- FIX: Pass image bytes wrapped in types.Image for RawReferenceImage ---
                raw_ref_bytes = tensor_to_bytes(images[0])
                raw_ref_image_type = types.Image(image_bytes=raw_ref_bytes)

                raw_ref = RawReferenceImage(
                    reference_image=raw_ref_image_type,
                    reference_id=0,
                )

                ref_images = [raw_ref]

                # Scribble Logic
                if scribble_mask is not None:
                     scribble_pil = tensor_to_pil(scribble_mask[0])
                     
                     # Ensure it has solid background if Model requires it. Google Docs say mask should be scribble. 
                     # Usually scribble models accept black BG with white lines, but Imagen 3 'scribble' control 
                     # actually says it expects a sketch. For now, since user says red lines on transparent,
                     # we pass it as a PNG (which supports transparency).
                     buffered_scribble = io.BytesIO()
                     scribble_pil.save(buffered_scribble, format="PNG")
                     scribble_bytes = buffered_scribble.getvalue()
                     
                     scribble_ref_image_type = types.Image(image_bytes=scribble_bytes)

                     scribble_ref = ControlReferenceImage(
                        reference_id=2, # Using ID 2 to not conflict with Mask's ID 1
                        reference_image=scribble_ref_image_type,
                        config=ControlReferenceConfig(
                            control_type=ControlReferenceType.CONTROL_TYPE_SCRIBBLE,
                            enable_control_image_computation=False # User provided scribble
                        ),
                     )
                     ref_images.append(scribble_ref)

                # Mask Logic
                if mask is not None:
                     mask_pil = tensor_to_pil(mask[0])
                     # Ensure single channel mask handling if needed, but for bytes just convert to PNG
                     if mask.shape[-1] == 1:
                         mask_pil = Image.fromarray((mask[0,:,:,0].cpu().numpy() * 255).astype(np.uint8), mode='L')
                     
                     buffered_mask = io.BytesIO()
                     mask_pil.save(buffered_mask, format="PNG")
                     mask_bytes = buffered_mask.getvalue()
                     
                     mask_ref_image_type = types.Image(image_bytes=mask_bytes)

                     mask_mode = "MASK_MODE_USER_PROVIDED"
                     mask_ref = MaskReferenceImage(
                        reference_id=1,
                        reference_image=mask_ref_image_type,
                        config=MaskReferenceConfig(
                            mask_mode=mask_mode,
                            mask_dilation=0.01,
                        ),
                    )
                     ref_images.append(mask_ref)
                elif operation == "BACKGROUND_SWAP":
                    # Auto BG
                     mask_ref = MaskReferenceImage(
                        reference_id=1,
                        reference_image=None,
                        config=MaskReferenceConfig(
                            mask_mode="MASK_MODE_BACKGROUND",
                        ),
                    )
                     ref_images.append(mask_ref)
                
                # Edit Config
                edit_mode_map = {
                    "INPAINT_INSERTION": "EDIT_MODE_INPAINT_INSERTION",
                    "INPAINT_REMOVAL": "EDIT_MODE_INPAINT_REMOVAL",
                    "OUTPAINT": "EDIT_MODE_OUTPAINT",
                    "BACKGROUND_SWAP": "EDIT_MODE_BGSWAP"
                }
                
                edit_mode = edit_mode_map.get(operation)
                
                print(f"Calling client.models.edit_image with mode {edit_mode}...")
                response = client.models.edit_image(
                    model=model,
                    prompt=prompt,
                    reference_images=ref_images,
                    config=EditImageConfig(
                        edit_mode=edit_mode,
                        number_of_images=1
                    ),
                )
                
                if response.generated_images:
                    # FIX: Vertex SDK v1beta1 returns a list of objects where .image is a types.Image
                    # We need to extract bytes from it.
                    try:
                        first_image = response.generated_images[0].image
                        # Check if it has 'image_bytes' (types.Image) or is a PIL Image (unlikely but possible in some SDK versions)
                        if hasattr(first_image, "image_bytes"):
                             img_data = first_image.image_bytes
                             out_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
                        elif hasattr(first_image, "_image_bytes"): # Internal attribute fallback
                             img_data = first_image._image_bytes
                             out_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
                        elif isinstance(first_image, Image.Image):
                            out_pil = first_image
                        else:
                             # Last ditch: try to cast to bytes or string? No, just print type
                             print(f"[NanoBananaPro] Unknown image type in response: {type(first_image)}")
                             # Try accessing it as bytes if it behaves like that
                             try:
                                out_pil = Image.open(io.BytesIO(first_image)).convert("RGB")
                             except:
                                return (torch.zeros((1, 64, 64, 3)), f"Unknown Image Type: {type(first_image)}")

                        img_np = np.array(out_pil).astype(np.float32) / 255.0
                        final_tensor = torch.from_numpy(img_np).unsqueeze(0)
                        return (final_tensor, "Success (Vertex AI)")
                    except Exception as e:
                         print(f"[NanoBananaPro] Error processing response image: {e}")
                         return (torch.zeros((1, 64, 64, 3)), f"Error processing response: {e}")
                else:
                    return (torch.zeros((1, 64, 64, 3)), "No images returned from Vertex AI.")

            except Exception as e:
                print(f"[NanoBananaPro] SDK Error: {e}")
                return (torch.zeros((1, 64, 64, 3)), f"Vertex SDK Error: {e}")

        # 2. GENERATION (Text-to-Image)
        else: # operation == "GENERATE"
            
            # A) Vertex AI Path (JSON Key) - For Imagen models
            if client:
                print(f"Generating via Vertex AI SDK (Model: {model})...")
                
                # --- FIX: Gemini Generation using generate_content ---
                if model in ["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"]:
                    print(f"[NanoBananaPro] using generate_content for {model}")
                    try:
                         # 1. Prepare Content list
                         message_parts = [types.Part.from_text(text=prompt)]
                         
                         # 2. Add Images and Scribbles
                         added_count = 0
                         if images is not None:
                             for i in range(images.shape[0]):
                                 # If scribble_mask is provided, send as separate layers
                                 if scribble_mask is not None:
                                     print(f"[NanoBananaPro] Sending base image and scribble mask as separate layers (SDK).")
                                     # 1. Base Image
                                     base_bytes, base_mime = tensor_to_bytes(images[i])
                                     part_base = types.Part(inline_data=types.Blob(mime_type=base_mime, data=base_bytes))
                                     message_parts.append(part_base)
                                     
                                     # 2. Scribble Layer
                                     s_idx = min(i, scribble_mask.shape[0] - 1)
                                     s_bytes, s_mime = tensor_to_bytes(scribble_mask[s_idx])
                                     part_scribble = types.Part(inline_data=types.Blob(mime_type=s_mime, data=s_bytes))
                                     message_parts.append(part_scribble)
                                     
                                     added_count += 2
                                 else:
                                     img_bytes, mime_type = tensor_to_bytes(images[i])
                                 
                                     part = types.Part(
                                         inline_data=types.Blob(
                                             mime_type=mime_type,
                                             data=img_bytes
                                         )
                                     )
                                     message_parts.append(part)
                                     added_count += 1
                         elif scribble_mask is not None: # Just scribble
                             for i in range(scribble_mask.shape[0]):
                                 img_bytes, mime_type = tensor_to_bytes(scribble_mask[i])
                                 part = types.Part(
                                     inline_data=types.Blob(
                                         mime_type=mime_type,
                                         data=img_bytes
                                     )
                                 )
                                 message_parts.append(part)
                                 added_count += 1

                         print(f"[NanoBananaPro] Added {added_count} composite image(s) to request.")

                         
                         try:
                             response = client.models.generate_content(
                                 model=model,
                                 contents=message_parts,
                                 config=types.GenerateContentConfig(
                                     response_modalities=["IMAGE", "TEXT"] if response_modalities == "IMAGE+TEXT" else ["IMAGE"],
                                     thinking_config=types.ThinkingConfig(thinking_level="High" if thinking else "minimal", include_thoughts=thinking) if (model in ["gemini-3.1-flash-image-preview", "gemini-3-pro-image-preview"] and hasattr(types, "ThinkingConfig") and thinking) else None,
                                     image_config=types.ImageConfig(
                                         aspect_ratio=aspect_ratio,
                                         image_size=resolution
                                     )
                                 ),
                             )
                         except errors.APIError as api_err:
                             if api_err.code == 503 and final_api_key:
                                 print(f"[NanoBananaPro] Vertex AI returned 503 Service Unavailable for Gemini 3 Pro {aspect_ratio}. Falling back to public Google AI Studio endpoint...")
                                 try:
                                     # Temporarily disable vertex AI override for this client instance
                                     if "GOOGLE_GENAI_USE_VERTEXAI" in os.environ:
                                         del os.environ["GOOGLE_GENAI_USE_VERTEXAI"]
                                         
                                     fallback_client = genai.Client(api_key=final_api_key)
                                     response = fallback_client.models.generate_content(
                                         model=model,
                                         contents=message_parts,
                                         config=types.GenerateContentConfig(
                                             response_modalities=["IMAGE", "TEXT"] if response_modalities == "IMAGE+TEXT" else ["IMAGE"],
                                             thinking_config=types.ThinkingConfig(thinking_level="High" if thinking else "minimal", include_thoughts=thinking) if (model in ["gemini-3.1-flash-image-preview", "gemini-3-pro-image-preview"] and hasattr(types, "ThinkingConfig") and thinking) else None,
                                             image_config=types.ImageConfig(
                                                 aspect_ratio=aspect_ratio,
                                                 image_size=resolution
                                             )
                                         ),
                                     )
                                     # Restore Vertex AI env for future runs if we were using it
                                     if using_vertex:
                                         os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
                                         
                                 except Exception as fallback_e:
                                     # Restore Vertex AI env
                                     if using_vertex:
                                            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
                                     print(f"[NanoBananaPro] AI Studio Fallback Exception: {fallback_e}")
                                     import traceback
                                     traceback.print_exc()
                                     return (torch.zeros((1, 64, 64, 3)), f"Gemini 3 Pro Fallback Error: {fallback_e}")
                             else:
                                 # Re-raise if it's not a 503 or we have no fallback key
                                 raise api_err
                         
                         out_tensors = []
                         out_text = ""
                         
                         if response.candidates:
                             for part in response.candidates[0].content.parts:
                                 if part.text:
                                     out_text += part.text + "\n"
                                 if part.inline_data:
                                     img = Image.open(io.BytesIO(part.inline_data.data))
                                     img_np = np.array(img).astype(np.float32) / 255.0
                                     out_tensors.append(torch.from_numpy(img_np))
                         
                         if out_tensors:
                             return (torch.stack(out_tensors, dim=0), out_text)
                         else:
                             return (torch.zeros((1, 64, 64, 3)), f"No images generated. Text: {out_text}")

                    except Exception as e:
                        print(f"[NanoBananaPro] Gemini 3 Pro SDK Exception: {e}")
                        import traceback
                        traceback.print_exc()
                        return (torch.zeros((1, 64, 64, 3)), f"Gemini 3 Pro Generation Error: {e}")


                if "gemini" in model:
                     return (torch.zeros((1, 64, 64, 3)), "Warning: You provided a Vertex JSON Key but selected a Gemini model. Please select 'imagen-3.0-generate-001' for Vertex generation.")
                
                try:
                    response = client.models.generate_images(
                        model=model,
                        prompt=prompt,
                        config=types.GenerateImagesConfig(
                            number_of_images=1,
                            aspect_ratio=aspect_ratio if aspect_ratio else "1:1"
                        )
                    )
                    if response.generated_images:
                        out_pil = response.generated_images[0].image
                        img_np = np.array(out_pil).astype(np.float32) / 255.0
                        final_tensor = torch.from_numpy(img_np).unsqueeze(0)
                        return (final_tensor, "Success (Vertex AI Generation)")
                    else:
                        return (torch.zeros((1, 64, 64, 3)), "No images returned.")
                except Exception as e:
                    return (torch.zeros((1, 64, 64, 3)), f"Vertex Generation Error: {e}")

            # B) Legacy/Rest Path (String API Key) - For Gemini models
            if not final_api_key:
                 return (torch.zeros((1, 64, 64, 3)), "API Key is required.")
            
            # ... Fallback into REST code below ...

            # Standard REST Implementation (Kept for Gemini models using API Key directly)
            # URL
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={final_api_key}"
            
            parts = [{"text": prompt}]
            
            # Images and Scribbles (for Image-to-Image generation, not editing)
            if images is not None:
                for i in range(images.shape[0]):
                    if scribble_mask is not None:
                        print(f"[NanoBananaPro] Sending base image and scribble mask as separate layers (REST).")
                        # 1. Base Image
                        base_b64, base_mime = tensor_to_b64(images[i])
                        parts.append({"inlineData": {"mimeType": base_mime, "data": base_b64}})
                        
                        # 2. Scribble Layer
                        s_idx = min(i, scribble_mask.shape[0] - 1)
                        s_b64, s_mime = tensor_to_b64(scribble_mask[s_idx])
                        parts.append({"inlineData": {"mimeType": s_mime, "data": s_b64}})
                    else:
                        img_b64, mime_type = tensor_to_b64(images[i])
                        parts.append({
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": img_b64
                            }
                        })
            elif scribble_mask is not None:
                for i in range(scribble_mask.shape[0]):
                    img_b64, mime_type = tensor_to_b64(scribble_mask[i])
                    parts.append({
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": img_b64
                        }
                    })
             
            # Files
            if files:
                file_paths = [f.strip() for f in files.split('\n') if f.strip()]
                for fp in file_paths:
                    if os.path.exists(fp):
                        mime_type = "application/pdf" if fp.lower().endswith(".pdf") else "text/plain"
                        try:
                            with open(fp, "rb") as f:
                                b64_data = base64.b64encode(f.read()).decode("utf-8")
                                parts.append({
                                    "inlineData": {
                                        "mimeType": mime_type,
                                        "data": b64_data
                                    }
                                })
                        except: pass

            generation_config = {
                "responseModalities": [response_modalities] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"],
            }
            if thinking and model in ["gemini-3.1-flash-image-preview", "gemini-3-pro-image-preview"]:
                generation_config["thinkingConfig"] = {
                    "thinkingLevel": "High",
                    "includeThoughts": True
                }
            if aspect_ratio:
                 generation_config["imageConfig"] = {"aspectRatio": aspect_ratio}

            payload = {
                "contents": [{"parts": parts, "role": "user"}],
                "generationConfig": generation_config
            }
            if system_prompt:
                payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

            # Request
            try:
                print(f"Sending request to {url}...")
                headers = {'Content-Type': 'application/json'}
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                
                print(f"Response Code: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"API Error Body: {response.text}")
                    return (torch.zeros((1, 64, 64, 3)), f"API Error {response.status_code}: {response.text}")

                response_json = response.json()
                
                output_images = []
                output_texts = []
                
                candidates = response_json.get("candidates", [])
                for candidate in candidates:
                    content = candidate.get("content", {})
                    c_parts = content.get("parts", [])
                    for part in c_parts:
                        if "text" in part:
                            output_texts.append(part["text"])
                        if "inlineData" in part: 
                            b64_data = part["inlineData"]["data"]
                            img_data = base64.b64decode(b64_data)
                            img = Image.open(io.BytesIO(img_data)).convert("RGB")
                            output_images.append(img)
                        elif "inline_data" in part:
                            b64_data = part["inline_data"]["data"]
                            img_data = base64.b64decode(b64_data)
                            img = Image.open(io.BytesIO(img_data)).convert("RGB")
                            output_images.append(img)

                final_text = "\n".join(output_texts)
                
                if output_images:
                    print(f"Successfully processed {len(output_images)} images.")
                    tensors = []
                    for img in output_images:
                        img_np = np.array(img).astype(np.float32) / 255.0
                        tensors.append(torch.from_numpy(img_np))
                    final_image_tensor = torch.stack(tensors, dim=0)
                else:
                    print("No images found in response.")
                    final_image_tensor = torch.zeros((1, 64, 64, 3))
                    if not final_text:
                        final_text = "No content generated."

                return (final_image_tensor, final_text)

            except Exception as e:
                print(f"Exception during request/parsing: {e}")
                return (torch.zeros((1, 64, 64, 3)), f"Exception: {e}")

class GeminiVeo31VideoGenerator:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic shot of...",
                    "tooltip": "The text description of the video you want to generate."
                }),
                "model": (["veo-2.0-generate-preview", "veo-2.0-generate-exp", "veo-3.1-generate-preview", "veo-3.1-fast-generate-preview"], {
                    "default": "veo-3.1-generate-preview",
                    "tooltip": "Select the Veo model. (Note: veo-2.0 is required for inpainting/mask operations in some API versions)."
                }),
                "mode": (["text_to_video", "image_to_video", "first_last_frame", "extend_video", "inpaint_insertion", "inpaint_removal", "reference images"], {
                    "default": "text_to_video",
                    "tooltip": "Select the generation mode."
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text describing what not to include."
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Gemini API Key",
                    "tooltip": "Your Google AI Studio API Key. Required."
                }),
                "service_account_json": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Path to JSON Key (Vertex AI)",
                    "tooltip": "Absolute path to your Vertex AI Service Account JSON key file. Required for Veo 2.0 Inpainting."
                }),
                "gcs_bucket": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Google Cloud Storage Bucket Name",
                    "tooltip": "Name of your Cloud Storage bucket (e.g. 'my-veo-bucket'). Required for Veo 2.0 Inpainting."
                }),
                "image": ("IMAGE", {"tooltip": "Input image (for image_to_video, first frame of FF2LF, or reference images for txt2v)."}),
                "last_frame": ("IMAGE", {"tooltip": "Ending frame. Used only in first_last_frame mode."}),
                "mask": ("MASK", {"tooltip": "Mask image for inpaint_insertion and inpaint_removal modes (white=edit, black=keep)."}),
                "video": ("VIDEO", {
                    "forceInput": True,
                    "tooltip": "Connect a Load Video output for inpainting or extending."
                }),
                "video_extend_in": ("STRING", {
                    "forceInput": True,
                    "default": "",
                    "tooltip": "Connect the video_extend (URI string) from a previous Veo node to extend it."
                }),
                "aspect_ratio": (["16:9", "9:16"], {
                    "default": "16:9",
                    "tooltip": "The aspect ratio of the generated video."
                }),
                "resolution": (["720p", "1080p", "4k"], {
                    "default": "720p",
                    "tooltip": "Resolution. Note: 1080p and 4k forced to 8s duration. Extension is 720p only."
                }),
                "duration": (["4", "6", "8"], {
                    "default": "4",
                    "tooltip": "Duration in seconds. Forced to 8s if 1080p/4k, or if using reference images."
                }),
                "person_generation": (["allow_all", "allow_adult", "dont_allow"], {
                    "default": "allow_all",
                    "tooltip": "Control person generation."
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed (improves determinism but not guaranteed)."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_video", "video_extend")
    FUNCTION = "generate"
    CATEGORY = "3DELAB"
    COLOR = "#940000"

    def upload_file_to_gemini(self, file_path, api_key):
        """Uploads a local file to Gemini File API and returns the URI."""
        try:
            mime_type = "video/mp4"
            if file_path.lower().endswith(".webm"):
                mime_type = "video/webm"
            elif file_path.lower().endswith(".mov"):
                mime_type = "video/quicktime"
            
            print(f"Uploading {file_path} to Gemini File API...")
            headers = {"x-goog-api-key": api_key}
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, mime_type)}
                resp = requests.post("https://generativelanguage.googleapis.com/upload/v1beta/files", headers=headers, files=files)
            
            if resp.status_code == 200:
                uri = resp.json().get("file", {}).get("uri")
                print(f"Successfully uploaded. URI: {uri}")
                return uri, mime_type
            else:
                print(f"File upload failed: {resp.status_code} {resp.text}")
                return None, None
        except Exception as e:
            print(f"Exception during file upload: {e}")
            return None, None

    def upload_file_to_gcs(self, file_path, bucket_name):
        """Uploads a local file to a Google Cloud Storage bucket and returns the gs:// URI."""
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(bucket_name.strip())
            
            blob_name = f"nano_banana_pro_veo_{int(time.time())}_{os.path.basename(file_path)}"
            blob = bucket.blob(blob_name)
            
            mime_type = "video/mp4"
            if file_path.lower().endswith(".webm"):
                mime_type = "video/webm"
            elif file_path.lower().endswith(".mov"):
                mime_type = "video/quicktime"
            elif file_path.lower().endswith(".png"):
                mime_type = "image/png"
            
            print(f"Uploading {file_path} to Google Cloud Storage: gs://{bucket_name}/{blob_name} ...")
            blob.upload_from_filename(file_path, content_type=mime_type)
            
            gs_uri = f"gs://{bucket.name}/{blob.name}"
            print(f"Successfully uploaded to GCS. URI: {gs_uri}")
            return gs_uri, mime_type
        except ImportError:
             print("Error: 'google-cloud-storage' library is required for Veo 2.0. run: pip install google-cloud-storage")
             return None, None
        except Exception as e:
             print(f"Exception during GCS upload: {e}")
             return None, None

    def resolve_video_input(self, video_input, video_extend_in=""):
        """Resolves the unified video input to either a local filepath or an existing URI."""
        if video_extend_in and getattr(video_extend_in, "startswith", lambda x: False)("https://") or getattr(video_extend_in, "startswith", lambda x: False)("gs://"):
            return None, video_extend_in
            
        if not video_input:
            return None, None

        # 1. Local File Path directly (String)
        if hasattr(video_input, "startswith") and os.path.exists(video_input):
            return video_input, None
            
        # 2. ComfyUI Native VIDEO Object (InputImpl.VideoFromFile logic)
        if hasattr(video_input, "path") and os.path.exists(getattr(video_input, "path", "")):
             return video_input.path, None
             
        # 3. ComfyUI hidden attributes (sometimes objects wrap it as _video or _path)
        if hasattr(video_input, "_path") and os.path.exists(getattr(video_input, "_path", "")):
             return video_input._path, None
             
        # 4. Comfy API v1 latest (it stores as __file, which Python mangles to _VideoFromFile__file)
        if hasattr(video_input, "_VideoFromFile__file"):
             p = getattr(video_input, "_VideoFromFile__file")
             if isinstance(p, str) and os.path.exists(p): return p, None
             
        # 5. Another variant: Some video objects might use get_file_path()
        if hasattr(video_input, "get_file_path"):
             try:
                 p = video_input.get_file_path()
                 if os.path.exists(p): return p, None
             except: pass

        # 6. Dictionary object (Alternative way some custom nodes pass video)
        if isinstance(video_input, dict) and "path" in video_input and os.path.exists(video_input["path"]):
             return video_input["path"], None
             
        # 7. List/Tuple containing a path or object
        if isinstance(video_input, (list, tuple)) and len(video_input) > 0:
             return self.resolve_video_input(video_input[0], video_extend_in)
             
        # 8. Fallback: Parse string representation directly (e.g. <VideoFromFile 'C:\...'>)
        vid_str = str(video_input)
        if "VideoFromFile" in vid_str or "Video" in vid_str:
            # Extract everything between quotes
            import re
            matches = re.findall(r"['\"](.*?)['\"]", vid_str)
            for m in matches:
                if os.path.exists(m):
                    print(f"NanoBananaPro parsed hidden path: {m}")
                    return m, None
             
        print(f"Warning: Could not parse video input format: {type(video_input)} - Value: {vid_str}")
        return None, None

    def generate(self, prompt, model, mode, negative_prompt="", api_key="", service_account_json="", gcs_bucket="", image=None, last_frame=None, mask=None, video=None, video_extend_in="", aspect_ratio="16:9", resolution="720p", duration="4", person_generation="allow_all", seed=42):
        print(f"--- [NanoBananaPro] Starting Veo 3.1/2.0: {mode} ---")
        
        # Early Validation for Mask
        if mode in ["inpaint_insertion", "inpaint_removal"]:
            if mask is None:
                print("Error: Inpaint modes require a mask input.")
                return ("Error: 'mask' input is required for inpaint modes. Please connect a mask.", "")
            if torch.max(mask) == 0.0:
                print("Error: The mask tensor is completely empty (no white pixels).")
                return ("Error: You connected a mask, but it's completely empty. Right-click the Image node -> 'Open in MaskEditor', draw your mask, and 'Save to node' before queuing.", "")

        api_key = api_key.strip()
        service_account_json = service_account_json.strip()
        gcs_bucket = gcs_bucket.strip()
        
        # Read files if they are paths
        if api_key and os.path.isfile(api_key):
             try:
                with open(api_key, 'r') as f:
                    api_key = f.read().strip()
                print("Loaded API Key from file.")
             except Exception as e:
                 print(f"Error reading API key file: {e}")

        # Determine Route (Vertex vs Gemini API)
        using_vertex = False
        project_id = ""
        location = "us-central1"
        client = None

        if service_account_json and ((mode in ["inpaint_insertion", "inpaint_removal", "extend_video"] and model == "veo-2.0-generate-preview") or model == "veo-2.0-generate-exp"):
            if not os.path.exists(service_account_json):
                return (f"Error: JSON Key file not found at: {service_account_json}", "")
            if mode in ["inpaint_insertion", "inpaint_removal"] and not gcs_bucket:
                return ("Error: 'gcs_bucket' is required when using Vertex AI inpaint modes.", "")
            if not GOOGLE_GENAI_AVAILABLE:
                return ("Error: 'google-genai' library missing. Install it for Vertex AI.", "")
            
            try:
                with open(service_account_json, 'r') as f:
                    creds = json.load(f)
                    if "project_id" in creds:
                        project_id = creds["project_id"]
                        print(f"[NanoBananaPro] Auto-detected Project ID: {project_id}")
            except Exception as e:
                print(f"[NanoBananaPro] Warning: Could not parse Project ID from JSON: {e}")
            
            if not project_id:
                return ("Error: Project ID could not be determined from the service account JSON.", "")
            
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_json
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
            try:
                client = genai.Client(vertexai=True, project=project_id, location=location)
                using_vertex = True
                print("[NanoBananaPro] Configured for Vertex AI via python SDK.")
            except Exception as e:
                return (f"Error Initializing Vertex Client: {e}", "")
        elif not api_key and model != "veo-2.0-generate-exp":
             return ("Error: API Key is required when not using Vertex AI.", "")

        # Helper to convert Tensor [B, H, W, C] to Base64 (RGB/RGBA)
        def tensor_to_b64(tensor):
            i_np = (tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(i_np)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
            
        # Helper to convert Single Channel Mask Tensor [B, H, W] to Base64
        def mask_to_b64(tensor):
            i_np = (tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            if len(i_np.shape) == 3 and i_np.shape[0] == 1:
                i_np = i_np[0]
            img = Image.fromarray(i_np, mode='L')
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Helper to save Tensor locally for Vertex GCS Uploads
        def tensor_to_temp_file(tensor, prefix="img"):
            i_np = (tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            if len(i_np.shape) == 3 and i_np.shape[0] == 1:
                i_np = i_np[0] # Handle 1-channel masks
            mode = 'L' if len(i_np.shape) == 2 else 'RGB'
            img = Image.fromarray(i_np, mode=mode)
            temp_path = os.path.join(folder_paths.get_temp_directory(), f"nanobanana_tmp_{prefix}_{int(time.time())}.png")
            img.save(temp_path)
            return temp_path

        # If Using Vertex SDK (Inpainting Veo 2.0)
        if using_vertex:
            if mode in ["inpaint_insertion", "inpaint_removal"] and mask is None:
                return (f"Error: 'mask' node input is required for {mode} mode.", "")
            
            local_vid_path, existing_uri = self.resolve_video_input(video, video_extend_in)
            if mode in ["inpaint_insertion", "inpaint_removal", "extend_video"]:
                if not local_vid_path and not existing_uri:
                     return (f"Error: Valid 'video' or 'video_extend_in' connection required for Vertex {mode}.", "")
                
                video_gcs_uri = existing_uri
                if local_vid_path:
                    print(f"Vertex Mode: Uploading video {local_vid_path} to GCS...")
                    uploaded_uri, upload_mime = self.upload_file_to_gcs(local_vid_path, gcs_bucket)
                    if uploaded_uri:
                        video_gcs_uri = uploaded_uri
                    else:
                        return ("Error: Failed to upload video to GCS.", "")

            # If mode is extend, the config structure in Vertex is slightly different.
            if mode == "extend_video":
                 config = types.GenerateVideosConfig(
                        aspect_ratio=aspect_ratio,
                        person_generation=person_generation.upper() if person_generation != "allow_adult" else "ALLOW_ADULT",
                        output_gcs_uri=f"gs://{gcs_bucket}/outputs/", 
                 )
                 source = types.GenerateVideosSource(
                        prompt=prompt if prompt else "extend the video",
                        video=types.Video(uri=video_gcs_uri, mime_type="video/mp4")
                 )
            elif mode in ["inpaint_insertion", "inpaint_removal"]:
                # Mask needs to go to GCS too for Vertex Veo 2.0 Inpaint
                temp_mask_path = tensor_to_temp_file(mask[0], prefix="mask")
                mask_gcs_uri, _ = self.upload_file_to_gcs(temp_mask_path, gcs_bucket)
                if not mask_gcs_uri:
                     return ("Error: Failed to upload mask to GCS.", "")
                
                mask_mode = types.VideoGenerationMaskMode.INSERT if mode == "inpaint_insertion" else types.VideoGenerationMaskMode.REMOVE
                
                safe_prompt = prompt if prompt else "Remove the object" 
                
                config = types.GenerateVideosConfig(
                            mask=types.VideoGenerationMask(
                                image=types.Image(gcs_uri=mask_gcs_uri, mime_type="image/png"),
                                mask_mode=mask_mode,
                            ),
                            aspect_ratio=aspect_ratio,
                            person_generation=person_generation.upper() if person_generation != "allow_adult" else "ALLOW_ADULT",
                            output_gcs_uri=f"gs://{gcs_bucket}/outputs/", 
                )
                source = types.GenerateVideosSource(
                            prompt=safe_prompt,
                            video=types.Video(uri=video_gcs_uri, mime_type="video/mp4")
                )
            # Veo 2.0 Generation/Reference Images
            else:
                config_kwargs = {
                    "aspect_ratio": aspect_ratio,
                    "person_generation": person_generation.upper() if person_generation != "allow_adult" else "ALLOW_ADULT",
                }
                
                # gcs_bucket is optional for standard generation, Vertex will return bytes instead if omitted
                if gcs_bucket:
                     config_kwargs["output_gcs_uri"] = f"gs://{gcs_bucket}/outputs/"
                     
                config = types.GenerateVideosConfig(**config_kwargs)
                source_kwargs = {"prompt": prompt}
                
                if mode == "reference images" and image is not None:
                     # Upload images to temp files then to GCS for Vertex SDK processing
                     ref_images = []
                     for i in range(min(3, image.shape[0])):
                          temp_img_path = tensor_to_temp_file(image[i], prefix=f"ref_{i}")
                          # If gcs_bucket provided upload there, else local bytes logic
                          if gcs_bucket:
                               img_gcs_uri, _ = self.upload_file_to_gcs(temp_img_path, gcs_bucket)
                               if img_gcs_uri:
                                   ref_images.append(
                                        types.VideoGenerationReferenceImage(
                                             image=types.Image(gcs_uri=img_gcs_uri, mime_type="image/png"),
                                             reference_type="ASSET"
                                        )
                                   )
                          else:
                               # Send inline bytes via the types.Image class
                               with open(temp_img_path, "rb") as f:
                                   ref_images.append(
                                       types.VideoGenerationReferenceImage(
                                            image=types.Image(image_bytes=f.read()),
                                            reference_type="ASSET"
                                       )
                                   )
                     
                     if ref_images:
                          config.reference_images = ref_images
                          
                source = types.GenerateVideosSource(**source_kwargs)
            
            print("Sending generation request to Vertex AI Veo 2.0...")
            try:
                # Execute Vertex SDK Call
                operation = client.models.generate_videos(
                    model=model,
                    source=source,
                    config=config,
                )
                
                print("Vertex AI Processing began. Waiting for completion...")
                print("Vertex AI Processing began. Waiting for completion...")
                # In google-genai, if generate_videos() didn't block and returned an operation:
                # we wait. But wait, generate_videos() in google-genai v0.3 *blocks* until done by default!
                # If it returned, it might already be done or failed. Let's inspect the returned object safely.
                if hasattr(operation, "done") and not operation.done:
                    is_done = False
                    while not is_done:
                         time.sleep(15)
                         op_info = client.operations.get(operation=operation)
                         
                         if op_info.done:
                              is_done = True
                              operation = op_info # Final State
                              print(f"Vertex SDK Polling: Done!")
                         else:
                              print(f"Status update: Polling operation details...")
                else:
                    print("Vertex SDK Polling: Done!")
                
                if hasattr(operation, 'error') and operation.error:
                    print(f"Vertex SDK Error Details: {operation.error}")
                    return (f"Vertex AI Error: {operation.error}", "")

                if hasattr(operation, 'result') and operation.result and hasattr(operation.result, 'generated_videos') and operation.result.generated_videos:
                    out_uri = "local_bytes"
                    # Download from GCS to ComfyUI Output directory
                    try:
                        out_uri = "local_bytes"
                        # Handle case where it returned GCS URI Check
                        if hasattr(operation.result.generated_videos[0], 'video') and hasattr(operation.result.generated_videos[0].video, 'uri'):
                            out_uri = operation.result.generated_videos[0].video.uri
                            print(f"Vertex AI Generation Complete! Output URI: {out_uri}")
                            
                            from google.cloud import storage
                            out_bucket_name = out_uri.split("gs://")[1].split("/")[0]
                            out_blob_name = out_uri.split(f"gs://{out_bucket_name}/")[1]
                            
                            dl_client = storage.Client()
                            dl_bucket = dl_client.bucket(out_bucket_name)
                            dl_blob = dl_bucket.blob(out_blob_name)
                            
                            output_dir = folder_paths.get_output_directory()
                            local_filename = f"veo20_{mode}_{int(time.time())}.mp4"
                            local_out_path = os.path.join(output_dir, local_filename)
                            
                            print(f"Downloading {out_uri} to {local_out_path}...")
                            dl_blob.download_to_filename(local_out_path)
                            print("Download complete!")
                            return (local_out_path, out_uri)

                        # Handle inline bytes if not GCS
                        elif hasattr(operation.result.generated_videos[0], 'video') and hasattr(operation.result.generated_videos[0].video, 'video_bytes'):
                            output_dir = folder_paths.get_output_directory()
                            local_filename = f"veo20_{mode}_{int(time.time())}.mp4"
                            local_out_path = os.path.join(output_dir, local_filename)
                            print(f"Saving returned inline bytes to {local_out_path}...")
                            with open(local_out_path, "wb") as f:
                                f.write(operation.result.generated_videos[0].video.video_bytes)
                            return (local_out_path, local_out_path)
                            
                            
                        else:
                             return ("Error: Could not parse video URI or bytes from Vertex response.", "")
                        
                    except Exception as dl_e:
                        print(f"Error downloading from Vertex response: {dl_e}")
                        return (out_uri, out_uri) # Fallback to string if download fails
                        
                else:
                    return (f"Vertex AI Error: {operation.error}", "")
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                return (f"Exception calling Vertex AI: {e}", "")

        # --- Standard REST API Execution (Veo 3.1) ---
        is_vertex_rest = False

        # Build constraints
        if resolution in ["1080p", "4k"] or mode == "extend_video":
            duration_val = 8
        else:
            duration_val = int(duration)

        parameters = {
            "aspectRatio": aspect_ratio,
            "resolution": resolution,
            "durationSeconds": duration_val,
            "personGeneration": person_generation,
            "seed": seed % 4294967296
        }
            
        if negative_prompt.strip() and mode != "inpaint_removal":
            parameters["negativePrompt"] = negative_prompt.strip()

        # Prompt is ignored/should be empty for pure removal, but required for insertion
        if mode == "inpaint_removal":
            instance = {"prompt": ""} 
        else:
            instance = {"prompt": prompt}

        # Handle Modes & Overrides
        if mode == "text_to_video":
            # If image is provided in txt2v, treat as reference images
            if image is not None:
                duration_val = 8
                parameters["durationSeconds"] = duration_val
                if person_generation == "allow_all":
                    parameters["personGeneration"] = "allow_adult"
                parameters["referenceImages"] = []
                for i in range(min(3, image.shape[0])):
                    img_b64 = tensor_to_b64(image[i])
                    parameters["referenceImages"].append({
                        "image": {
                            "bytesBase64Encoded": img_b64,
                            "mimeType": "image/png"
                        },
                        "referenceType": "ASSET"
                    })
                print(f"[NanoBananaPro] Added {len(parameters['referenceImages'])} reference images.")

        elif mode == "image_to_video":
            if image is None:
                return ("Error: 'image' input is required for image_to_video mode.", "")
            img_b64 = tensor_to_b64(image[0])
            instance["image"] = {
                "bytesBase64Encoded": img_b64,
                "mimeType": "image/png"
            }
            if person_generation == "allow_all":
                parameters["personGeneration"] = "allow_adult"

        elif mode in ["inpaint_insertion", "inpaint_removal"]:
            if mask is None:
                return (f"Error: 'mask' node input is required for {mode} mode.", "")
            
            local_vid_path, existing_uri = self.resolve_video_input(video, video_extend_in)
            video_uri_to_use = existing_uri
            video_mime = "video/mp4"

            if local_vid_path:
                print(f"Processing local video for inpainting from: {local_vid_path}")
                uploaded_uri, upload_mime = self.upload_file_to_gemini(local_vid_path, api_key)
                if uploaded_uri:
                    video_uri_to_use = uploaded_uri
                    video_mime = upload_mime
                else:
                    return ("Error: Failed to upload video to Gemini File API.", "")
                    
            if not video_uri_to_use:
                 # Fallback to single image if video input fails but image is provided
                 print("Warning: No valid 'video' input found. Falling back to static image processing for inpaint.")
                 if image is None:
                     return (f"Error: Valid 'video' connection or fallback 'image' required for {mode}.", "")
                 
                 img_b64 = tensor_to_b64(image[0])
                 # Vertex AI docs state video object can hold fileUri, but image holds bytesBase64Encoded.
                 instance["image"] = {
                     "bytesBase64Encoded": img_b64,
                     "mimeType": "image/png"
                 }
            else:
                # Based on Vertex AI API schema for Veo 2.0 inpainting
                # Note: The API uses gcsUri or fileUri depending on the exact endpoint version (Vertex vs Gemini)
                # Since we are using the Gemini API endpoint, we continue using fileData if that's what File API returns
                # or a simple structure if it matches the Veo 3.1 extend format.
                instance["video"] = {
                    "fileData": {
                        "fileUri": video_uri_to_use,
                        "mimeType": video_mime
                    }
                }
            
            mask_b64 = mask_to_b64(mask[0])
            mask_mode = "insert" if mode == "inpaint_insertion" else "remove"
            # As per Vertex AI payload example:
            # "mask": { "gcsUri": "...", "mimeType": "...", "maskMode": "insert" }
            # Since we have base64 from ComfyUI instead of a GCS URI:
            instance["mask"] = {
                "bytesBase64Encoded": mask_b64,
                "mimeType": "image/png",
                "maskMode": mask_mode
            }
            
            if person_generation == "allow_all":
                parameters["personGeneration"] = "allow_adult"

        elif mode == "reference images":
            if model not in ["veo-2.0-generate-exp", "veo-3.1-generate-preview"]:
                return (f"Error: 'reference images' mode is only supported with 'veo-2.0-generate-exp' and 'veo-3.1-generate-preview'. Current model is '{model}'.", "")
            if image is None:
                return ("Error: 'image' input is required for 'reference images' mode.", "")
            
            instance["referenceImages"] = []
            for i in range(min(3, image.shape[0])):
                img_b64 = tensor_to_b64(image[i])
                instance["referenceImages"].append({
                    "image": {
                        "bytesBase64Encoded": img_b64,
                        "mimeType": "image/png"
                    },
                    "referenceType": "asset"
                })
            print(f"[NanoBananaPro] Added {len(instance['referenceImages'])} reference images.")
            if person_generation == "allow_all":
                parameters["personGeneration"] = "allow_adult"

        elif mode == "first_last_frame":

            if image is None or last_frame is None:
               return ("Error: Both 'image' and 'last_frame' are required for first_last_frame mode.",)
            img_b64 = tensor_to_b64(image[0])
            last_img_b64 = tensor_to_b64(last_frame[0])
            
            instance["image"] = {
                "bytesBase64Encoded": img_b64,
                "mimeType": "image/png"
            }
            # Put lastFrame in instance, SDK puts it there surprisingly
            instance["lastFrame"] = {
                "bytesBase64Encoded": last_img_b64,
                "mimeType": "image/png"
            }
            
            if person_generation == "allow_all":
                parameters["personGeneration"] = "allow_adult"

        elif mode == "extend_video":
            local_vid_path, existing_uri = self.resolve_video_input(video, video_extend_in)
            
            # If the user passed a ComfyUI VIDEO, we can upload it to Gemini!
            if local_vid_path and not existing_uri:
                print(f"Processing local video for extending from: {local_vid_path}")
                uploaded_uri, upload_mime = self.upload_file_to_gemini(local_vid_path, api_key)
                if uploaded_uri:
                    existing_uri = uploaded_uri
                else:
                    return ("Error: Failed to upload local video for extending.", "")

            if not existing_uri or not existing_uri.startswith("https://"):
                return ("Error: Invalid video format for extending. Must be a previously generated Veo output or a valid Load Video.", "")
            
            # Veo 3.1 explicitly rejects fileData on the REST endpoint
            instance["video"] = {"uri": existing_uri}
            
            # Remove parameters that aren't allowed for extend_video in Gemini API
            parameters.pop("resolution", None)
            parameters.pop("durationSeconds", None)
            parameters["sampleCount"] = 1

        payload = {
            "instances": [instance],
            "parameters": parameters
        }

        print(f"Sending generation request to Veo 3.1 ({model})...")
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            if response.status_code != 200:
                print(f"API Error {response.status_code}: {response.text}")
                return (f"API Error {response.status_code}: {response.text}", "")
            
            resp_json = response.json()
            operation_name = resp_json.get("name")
            if not operation_name:
                return (f"Error: No operation name returned. {response.text}", "")
            
            print(f"Operation started: {operation_name}")
            
            # Polling
            if is_vertex_rest:
                # Vertex returns operation_name as "projects/.../locations/.../publishers/.../models/.../operations/..."
                # The correct polling endpoint for these LRO operations is just appending the operation name directly to the base URL
                # OR using the global operations endpoint. Let's try the operations endpoint specifically.
                # Actually, the base url is `https://{location}-aiplatform.googleapis.com/v1` and the returned name is the FULL relative path.
                import re
                poll_op_name = re.sub(r'publishers/[^/]+/models/[^/]+/', '', operation_name)
                poll_url = f"https://us-central1-aiplatform.googleapis.com/v1/{poll_op_name}"
            else:
                poll_url = f"{base_url}/{operation_name}"
                
            is_done = False
            video_uri = None
            
            while not is_done:
                print("Waiting for video generation to complete (10s)...")
                time.sleep(10)
                poll_resp = requests.get(poll_url, headers=headers)
                
                if poll_resp.status_code != 200:
                    print(f"Polling Error {poll_resp.status_code}: {poll_resp.text}")
                    return (f"Polling Error: {poll_resp.text}", "")
                
                poll_data = poll_resp.json()
                is_done = poll_data.get("done", False)
                
                if is_done:
                    if "error" in poll_data:
                        print(f"Generation Error: {poll_data['error']}")
                        return (f"Generation Error: {poll_data['error']}", "")
                    
                    try:
                        raw_video_uri = poll_data["response"]["generateVideoResponse"]["generatedSamples"][0]["video"]["uri"]
                        download_uri = raw_video_uri
                        video_uri = raw_video_uri
                        # Strip download suffix for extend_video compatibility
                        if ":download" in video_uri:
                            video_uri = video_uri.split(":download")[0]
                    except KeyError as e:
                        return (f"Error parsing successful response: {poll_data}", "")

            if video_uri:
                print(f"Video ready at URI: {video_uri}")
                # Generate unique filename
                h = hashlib.sha256(f"{prompt}_{time.time()}".encode()).hexdigest()[:8]
                out_filename = f"veo31_{mode}_{h}.mp4"
                out_path = os.path.join(self.output_dir, out_filename)
                
                print(f"Downloading video to {out_path}...")
                try:
                    # Append alt=media to actually request binary bytes instead of JSON metadata
                    dl_url = f"{download_uri}?alt=media" if "?" not in download_uri else f"{download_uri}&alt=media"
                    dl_resp = requests.get(dl_url, headers=headers)
                    
                    if dl_resp.status_code == 200:
                        with open(out_path, "wb") as f:
                            f.write(dl_resp.content)
                        print(f"Successfully saved to {out_path}")
                        return (out_path, video_uri)
                    else:
                        print(f"Download Error {dl_resp.status_code}: {dl_resp.text}")
                        return (f"Download Error {dl_resp.status_code}: {dl_resp.text}", video_uri)
                except Exception as e:
                    print(f"Failed to download video: {e}")
                    return (f"Failed to download video: {e}", video_uri)
                
            return ("Error: Polling finished but no video URI found.", "")

        except Exception as e:
            print(f"Exception during Veo 3.1 generation: {e}")
            import traceback
            traceback.print_exc()
            return (f"Exception: {e}", "")

class NanoBananaPreviewVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "3DELAB"
    COLOR = "#940000"

    def preview(self, video_path):
        import os
        import folder_paths
        
        if not video_path or not os.path.exists(video_path):
            return {"ui": {"video": []}}
            
        filename = os.path.basename(video_path)
        output_dir = folder_paths.get_output_directory()
        
        # Calculate subfolder if any
        subfolder = os.path.dirname(video_path).replace(output_dir, "").strip("\\/")
        
        return {
            "ui": {
                "video": [
                    {
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": "output"
                    }
                ]
            }
        }


class LoadVideoExtract:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {
            "required": {
                "video": (sorted(files), {"video_upload": True}),
                "frame_target": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1, "tooltip": "Frame number to extract and save."}),
                "save_prefix": ("STRING", {"default": "extracted_frame", "tooltip": "Prefix for the automatically saved image file."})
            }
        }

    CATEGORY = "3DELAB"
    COLOR = "#940000"
    RETURN_TYPES = ("VIDEO", "IMAGE")
    RETURN_NAMES = ("video", "extracted_image")
    FUNCTION = "load_video"

    def load_video(self, video, frame_target, save_prefix):
        video_path = folder_paths.get_annotated_filepath(video)
        
        # 1. Output VIDEO object
        try:
            from comfy_api.latest._input_impl.video_types import VideoFromFile
            video_out = VideoFromFile(video_path)
        except ImportError:
            video_out = video_path

        # 2. Extract specific frame using OpenCV and save to input folder
        image_tensor = torch.zeros((1, 64, 64, 3)) # Default fallback
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[NanoBananaPro] Error: Could not open video {video_path}")
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_target)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR (OpenCV) to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Instead of returning a tensor, we save it physically to ComfyUI's input dir
                    # so the user can use an Image node and open it in the Mask Editor.
                    input_dir = folder_paths.get_input_directory()
                    file_name = f"!_nano_{save_prefix}_{int(time.time())}.png"
                    save_path = os.path.join(input_dir, file_name)
                    
                    # Save as PNG
                    img_pil = Image.fromarray(frame_rgb)
                    img_pil.save(save_path)
                    
                    # Convert to normalized float32 tensor [1, H, W, C] for downstream nodes
                    img_np = np.array(frame_rgb).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(img_np)[None,]

                    # Notify UI to auto-select this file
                    try:
                        import server
                        server.PromptServer.instance.send_sync("nanobanana_video_extracted", {"filename": file_name})
                    except Exception as e:
                        print(f"[NanoBananaPro] Warning: Could not send UI sync message: {e}")
                    
                    print(f"[NanoBananaPro] Extracted frame {frame_target} successfully.")
                    print(f"[NanoBananaPro] Saved physical image to: {save_path}")
                else:
                    print(f"[NanoBananaPro] Warning: Could not read frame {frame_target} from {video_path}. It may exceed the video length.")
                cap.release()
        except ImportError:
            print("[NanoBananaPro] Error: OpenCV (cv2) is required for frame extraction. Install with: pip install opencv-python")
        except Exception as e:
            print(f"[NanoBananaPro] Error extracting frame: {e}")

        return (video_out, image_tensor)

class ImagePassthrough:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        exts = ['.png', '.jpg', '.jpeg', '.webp']
        files = [f for f in files if any(f.lower().endswith(ext) for ext in exts)]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
            "optional": {
                "image_in": ("IMAGE", {"tooltip": "If connected, this image overrides the selected file."}),
            }
        }

    CATEGORY = "3DELAB"
    COLOR = "#940000"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"

    @classmethod
    def VALIDATE_INPUTS(s, image, image_in=None):
        if image.startswith("clipspace/"):
            return True
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

    def load_image(self, image, image_in=None):
        image_path = folder_paths.get_annotated_filepath(image)
        try:
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            
            # Extract Mask from alpha channel (like standard LoadImage)
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - mask
                mask = torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
                
            image_rgb = i.convert("RGB")
            image_np = np.array(image_rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
        except Exception as e:
            print(f"[NanoBananaPro] Error loading image {image_path}: {e}")
            image_tensor = torch.zeros((1, 64, 64, 3))
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")

        if image_in is not None:
            print(f"[NanoBananaPro] ImagePassthrough: Using incoming image connection. Preserving mask from {image}.")
            return (image_in, mask.unsqueeze(0))

        print(f"[NanoBananaPro] ImagePassthrough: Loading file {image}")
        return (image_tensor, mask.unsqueeze(0))
