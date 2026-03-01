# ComfyUI Nano Banana Pro

This is a **custom node for ComfyUI** that integrates Google's **Gemini** and **Vertex AI (Imagen 3)** models for image generation and editing. It is a modified version of the original `nodes_gemini.py` node.

Este es un **nodo personalizado para ComfyUI** que integra los modelos **Gemini** y **Vertex AI (Imagen 3)** de Google para la generación y edición de imágenes. Es una versión modificada del nodo original `nodes_gemini.py`.

---

## 🇺🇸 English (v1.4.1)

### Features
- **Nano Banana 2 & Veo 3.1 Integration**: State-of-the-art models for image and video generation in ComfyUI.
- **Image Generation**: Supports `txt2img`, `img2img`, and up to **14 visual references** for `img2img`.
- **Image Editing (Imagen 3 / Vertex AI)**: Advanced tasks including Inpainting, Outpainting, and Background Swap.
- **Video Generation (Veo 3.1 / 2.0)**: Supports `txt2vid`, `img2vid`, and **NEW (v1.4.1): 3 reference `img2vid`** using Veo 2.0 via Vertex AI.
- **Video Editing (Vertex AI)**: Seamlessly Inpaint objects into videos, Remove objects, Extend clips, or generate First/Last frame interpolations.
- **Google Cloud Storage (GCS) Integration**: Auto-uploads local `.mp4` files and reference images directly to your specified GCS Bucket for Vertex AI tasks.
- **Utility Nodes**: Includes `Load Video (Extract Frame)` and `Load Image (Passthrough)` with JS automation (`!_nano_` prefix) to pipe auto-extracted frames straight into ComfyUI's native Mask Editor.
- **Load Image & Scribble Editor Node**: A custom node featuring an integrated HTML5 canvas to draw vector arrows, text, and raster scribbles.

### Authentication Requirements
This node uses a hybrid authentication system depending on the requested task:
- **API Key (Google AI Studio)**: Required for standard Image & Video Generation (`veo-3.1`, `gemini-3.1`).
- **Service Account JSON & GCS Bucket (Vertex AI)**: Required for advanced Image Editing (Imagen 3) and Veo 2.0 tasks (`veo-2.0-generate-exp`, Video Inpainting/Extension).

### Installation
1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```
2. Clone this repository:
   ```bash
   git clone https://github.com/PATAGONIABIM/comfyui-NanoBananaPro_3DeLAB.git
   ```
3. Install dependencies:
   ```bash
   cd comfyui-NanoBananaPro_3DeLAB
   pip install -r requirements.txt
   ```

### Configuration & Usage
To use the node, you need to provide authentication credentials in the node inputs:

1. **For Image Generation (Gemini Models):**
   - Obtain an API KEY from [Google AI Studio](https://aistudio.google.com/).
   - Paste the key into the `api_key` field or provide a path to a `.txt` file containing the key.

2. **For Image Editing (Imagen 3 / Vertex AI):**
   - You must have a Google Cloud Project with Vertex AI API enabled.
   - Create a Service Account and download the JSON key file.
   - Place the JSON file locally and provide the **absolute path** to it in the `service_account_json` field.
   - **Note:** Editing operations (Inpaint, Outpaint) *require* Vertex AI credentials.

### Core Parameters
- **prompt**: The text description of the image or video you want to generate.
- **model**: Select the AI model to use (Gemini, Imagen 3, or Veo variants).
- **operation**: Choose the operation mode for images: Generate, Inpaint, Outpaint, Background Swap... or mode for video: standard, inpaint, extend, reference images, etc.
- **api_key**: Your Google AI Studio API Key. Required for standard generation.
- **service_account_json**: Absolute path to your Vertex AI Service Account JSON key file. Required for editing and Veo 2.0.
- **gcs_bucket**: Name of your Google Cloud Storage Bucket. Required for uploading assets during Vertex AI video operations.
- **aspect_ratio**: The aspect ratio of the generated output (e.g., 1:1, 16:9).
- **images**: Input images for editing or references (up to 14 for images, up to 3 for Veo 2.0 video).
- **mask**: Mask image for inpainting (white = edit, black = keep).
- **scribble_mask**: Scribble or sketch image for controlled editing.
- **system_prompt**: System-level instructions to guide the model's behavior.

---

## 🇪🇸 Español (v1.4.1)

### Características
- **Integración Nano Banana 2 & Veo 3.1**: Modelos de última generación para creación de imágenes y video en ComfyUI.
- **Generación de Imágenes**: Soporta `txt2img`, `img2img`, y hasta **14 referencias visuales** para `img2img`.
- **Edición de Imágenes (Imagen 3 / Vertex AI)**: Tareas avanzadas incluyendo Inpainting, Outpainting y Cambio de Fondo.
- **Generación de Video (Veo 3.1 / 2.0)**: Soporta `txt2vid`, `img2vid` y **NUEVO (v1.4.1): 3 referencias para `img2vid`** usando Veo 2.0 vía Vertex AI.
- **Edición de Video (Vertex AI)**: Inserta objetos en videos de forma fluida (Inpaint), Remueve objetos, Extiende clips, o genera interpolación de Primer/Último frame.
- **Integración con Google Cloud Storage (GCS)**: Subida automática de archivos `.mp4` p locales e imágenes de referencia directamente al Bucket GCS especificado para las tareas de Vertex AI.
- **Nodos de Utilidad**: Incluye Nodos `Load Video (Extract Frame)` y `Load Image (Passthrough)` con automatización JS (prefijo `!_nano_`) para enviar frames extraídos automáticamente al Mask Editor nativo de ComfyUI.
- **Nodo Load Image & Scribble Editor**: Nodo personalizado que integra un Canvas HTML5 para dibujar flechas vectoriales, texto y trazos libres.

### Requisitos de Autenticación
Este nodo usa un sistema de autenticación híbrido dependiendo de la tarea solicitada:
- **API Key (Google AI Studio)**: Necesaria para la Generación estándar de Imágenes y Videos (`veo-3.1`, `gemini-3.1`).
- **Service Account JSON & GCS Bucket (Vertex AI)**: Necesarios para tareas avanzadas de Edición de Imágenes (Imagen 3) y tareas de Veo 2.0 (`veo-2.0-generate-exp`, Inpainting/Extensión de Video).

### Instalación
1. Navega a tu directorio de nodos personalizados de ComfyUI:
   ```bash
   cd ComfyUI/custom_nodes/
   ```
2. Clona este repositorio:
   ```bash
   git clone https://github.com/PATAGONIABIM/comfyui-NanoBananaPro_3DeLAB.git
   ```
3. Instala las dependencias:
   ```bash
   cd comfyui-NanoBananaPro_3DeLAB
   pip install -r requirements.txt
   ```

### Configuración y Uso
Para usar el nodo, debes proporcionar las credenciales de autenticación en las entradas del nodo:

1. **Para Generación de Imágenes (Modelos Gemini):**
   - Obtén una API KEY desde [Google AI Studio](https://aistudio.google.com/).
   - Pega la clave en el campo `api_key` o proporciona la ruta a un archivo `.txt` que contenga la clave.

2. **Para Edición de Imágenes (Imagen 3 / Vertex AI):**
   - Debes tener un Proyecto de Google Cloud con la API de Vertex AI habilitada.
   - Crea una Cuenta de Servicio (Service Account) y descarga el archivo de clave JSON.
   - Coloca el archivo JSON localmente y proporciona la **ruta absoluta** al mismo en el campo `service_account_json`.
   - **Nota:** Las operaciones de edición (Inpaint, Outpaint) *requieren* credenciales de Vertex AI.

### Parámetros Principales
- **prompt**: La descripción de texto de la imagen o video que deseas generar.
- **model**: Selecciona el modelo de IA a usar (variantes de Gemini, Imagen 3 o Veo).
- **operation**: Elige el modo de operación para imágenes: Generar, Inpaint, Outpaint, Cambio de Fondo... o modo para video: estándar, inpaint, extender, reference images, etc.
- **api_key**: Tu API Key de Google AI Studio. Requerida para generación estándar.
- **service_account_json**: Ruta absoluta a tu archivo JSON de Service Account de Vertex AI. Requerido para edición y Veo 2.0.
- **gcs_bucket**: Nombre de tu Bucket de Google Cloud Storage. Requerido para subir archivos durante las operaciones de video en Vertex AI.
- **aspect_ratio**: La relación de aspecto de la salida generada (ej. 1:1, 16:9).
- **images**: Imágenes de entrada para edición o referencias (hasta 14 para imágenes, hasta 3 para video en Veo 2.0).
- **mask**: Imagen de máscara para inpainting (blanco = editar, negro = mantener).
- **scribble_mask**: Imagen de garabato o boceto para edición controlada.
- **system_prompt**: Instrucciones a nivel de sistema para guiar el comportamiento del modelo.
