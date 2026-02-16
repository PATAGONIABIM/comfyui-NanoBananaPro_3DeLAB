import os
import torch
import numpy as np
from PIL import Image
import io
import base64
import requests
import json
import folder_paths
import random
import sys

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
        EditImageConfig,
        GenerateImagesConfig,
        GenerateContentConfig,
        Modality
    )
    from google.genai import types
    GOOGLE_GENAI_AVAILABLE = True
    print(f"[NanoBananaPro] google-genai successfully imported.")
except ImportError as e:
    GOOGLE_GENAI_AVAILABLE = False
    print(f"[NanoBananaPro] Failed to import google-genai: {e}")

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
                "model": (["gemini-3-pro-image-preview", "imagen-3.0-capability-001", "imagen-3.0-generate-001", "imagen-4.0-generate-001"], {
                    "default": "gemini-3-pro-image-preview",
                    "tooltip": "Select the AI model to use (Gemini 3 Pro or Imagen 3 variants)."
                }),
                "operation": (["GENERATE", "INPAINT_INSERTION", "INPAINT_REMOVAL", "OUTPAINT", "BACKGROUND_SWAP"], {
                    "default": "GENERATE",
                    "tooltip": "Choose the operation mode: GEN (Txt2Img), INPAINT (Edit), OUTPAINT (Expand), BG_SWAP."
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Gemini API Key (AI Studio) - For Gemini Models",
                    "tooltip": "Your Google AI Studio API Key. Required for Gemini models."
                }),
                "service_account_json": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Path to JSON Key (Vertex AI) - For Imagen Models",
                    "tooltip": "Absolute path to your Vertex AI Service Account JSON key file. Required for Imagen 3 editing/inpainting."
                }),
                "images": ("IMAGE", {"tooltip": "Input image for editing, inpainting, or image-to-image generation."}),
                "mask": ("MASK", {"tooltip": "Mask image for inpainting (white = edit, black = keep)."}),
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
                "files": ("STRING", {
                    "multiline": True, 
                    "placeholder": "Path to local file (PDF/TXT)",
                    "tooltip": "Path to local text or PDF files to use as context."
                }),
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

    def generate(self, prompt, model, operation, api_key="", service_account_json="", seed=None, aspect_ratio="1:1", resolution="1K", response_modalities="IMAGE", images=None, mask=None, files=None, system_prompt=""):
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
        
        # 1.5Force Global for Gemini 3 Pro (Experimental)
        if model == "gemini-3-pro-image-preview":
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

        # 1. Try Vertex AI (Service Account)
        if service_account_json:
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
        
        # 2. Try Gemini (API Key) logic is handled in fallback block or check here if strictly required
        elif api_key:
             # Legacy API Key Logic (File or String)
            if os.path.isfile(api_key):
                 try:
                    with open(api_key, 'r') as f:
                        final_api_key = f.read().strip()
                    print("Loaded API Key from file.")
                 except: 
                     pass
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
        # Helper to convert Tensor to B64
        def tensor_to_b64(tensor): # [H, W, C]
            img = tensor_to_pil(tensor)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Helper to convert Tensor to Bytes
        def tensor_to_bytes(tensor): # [H, W, C]
            img = tensor_to_pil(tensor)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return buffered.getvalue()

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
                
                # --- FIX: Gemini 3 Pro Generation using generate_content ---
                if model == "gemini-3-pro-image-preview":
                    print(f"[NanoBananaPro] using generate_content for {model}")
                    try:
                         # 1. Prepare Content Parts
                         message_parts = [types.Part.from_text(text=prompt)]
                         
                         # 2. Add Images if present
                         if images is not None:
                             for i in range(images.shape[0]):
                                 img_bytes = tensor_to_bytes(images[i])
                                 start_len = len(img_bytes)
                                 # Create Part from Bytes
                                 # Using types.Part(inline_data=...) pattern for safety
                                 part = types.Part(
                                     inline_data=types.Blob(
                                         mime_type="image/png",
                                         data=img_bytes
                                     )
                                 )
                                 message_parts.append(part)
                             print(f"[NanoBananaPro] Added {images.shape[0]} images to request.")

                         response = client.models.generate_content(
                            model=model,
                            contents=message_parts,
                            config=GenerateContentConfig(
                                response_modalities=[Modality.IMAGE, Modality.TEXT] if response_modalities == "IMAGE+TEXT" else [Modality.IMAGE],
                                safety_settings=[],
                            ),
                        )
                         
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
                        return (torch.zeros((1, 64, 64, 3)), f"Gemini 3 Pro Generation Error: {e}")


                if "gemini" in model:
                     return (torch.zeros((1, 64, 64, 3)), "Warning: You provided a Vertex JSON Key but selected a Gemini model. Please select 'imagen-3.0-generate-001' for Vertex generation.")
                
                try:
                    response = client.models.generate_images(
                        model=model,
                        prompt=prompt,
                        config=GenerateImagesConfig(
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
            
            # Images (for Image-to-Image generation, not editing)
            if images is not None:
                for i in range(images.shape[0]):
                    parts.append({
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": tensor_to_b64(images[i])
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
