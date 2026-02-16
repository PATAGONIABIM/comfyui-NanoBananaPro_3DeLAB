# ComfyUI Nano Banana Pro

This is a **custom node for ComfyUI** that integrates Google's **Gemini** and **Vertex AI (Imagen 3)** models for image generation and editing. It is a modified version of the original `nodes_gemini.py` node.

Este es un **nodo personalizado para ComfyUI** que integra los modelos **Gemini** y **Vertex AI (Imagen 3)** de Google para la generación y edición de imágenes. Es una versión modificada del nodo original `nodes_gemini.py`.

---

## 🇺🇸 English

### Features
- **Unlocks Gemini 3 Pro** via `google-genai` library (replacing standard REST API).
- **Supports Imagen 3** for advanced editing tasks (Inpainting, Outpainting, Background Swap).
- Hybrid 1:1 Authentication:
  - **API Key (Google AI Studio)**: Required for standard Image Generation.
  - **Service Account JSON (Vertex AI)**: Required for Image Editing and Inpainting operations.

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
- **prompt**: The text description of the image you want to generate.
- **model**: Select the AI model to use (Gemini 3 Pro or Imagen 3 variants).
- **operation**: Choose the operation mode: Generate, Inpaint (Insert/Remove), Outpaint, or Background Swap.
- **api_key**: Your Google AI Studio API Key. Required for Gemini models.
- **service_account_json**: Absolute path to your Vertex AI Service Account JSON key file. Required for Imagen 3 editing/inpainting.
- **aspect_ratio**: The aspect ratio of the generated image (e.g., 1:1, 16:9).
- **images**: Input image for editing, inpainting, or image-to-image generation.
- **mask**: Mask image for inpainting (white = edit, black = keep).
- **seed**: Seed for random number generation. Use -1 or slightly change for variation.
- **resolution**: Resolution of the output image (1K, 2K, 4K).
- **response_modalities**: Choose strictly image output or image + text reasoning (if supported).
- **files**: Path to local text or PDF files to use as context.
- **system_prompt**: System-level instructions to guide the model's behavior.

---

## 🇪🇸 Español

### Características
- **Desbloquea Gemini 3 Pro** a través de la librería `google-genai` (reemplazo de API REST estándar).
- **Soporta Imagen 3** para tareas avanzadas de edición (Inpainting, Outpainting, Cambio de Fondo).
- Autenticación Híbrida 1:1:
  - **API Key (Google AI Studio)**: Necesaria para la Generación de Imágenes estándar.
  - **Service Account JSON (Vertex AI)**: Necesaria para operaciones de Edición de Imágenes e Inpainting.

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
- **prompt**: La descripción de texto de la imagen que deseas generar.
- **model**: Selecciona el modelo de IA a usar (Gemini 3 Pro o variantes de Imagen 3).
- **operation**: Elige el modo de operación: Generar, Inpaint (Insertar/Eliminar), Outpaint o Cambio de Fondo.
- **api_key**: Tu API Key de Google AI Studio. Requerida para modelos Gemini.
- **service_account_json**: Ruta absoluta a tu archivo de clave JSON de Service Account de Vertex AI. Requerido para edición/inpainting con Imagen 3.
- **aspect_ratio**: La relación de aspecto de la imagen generada (ej. 1:1, 16:9).
- **images**: Imagen de entrada para edición, inpainting o generación imagen-a-imagen.
- **mask**: Imagen de máscara para inpainting (blanco = editar, negro = mantener).
- **seed**: Semilla para generación de números aleatorios. Usa -1 o cambia ligeramente para variaciones.
- **resolution**: Resolución de la imagen de salida (1K, 2K, 4K).
- **response_modalities**: Elige salida estrictamente de imagen o imagen + texto de razonamiento (si está soportado).
- **files**: Ruta a archivos de texto o PDF locales para usar como contexto.
- **system_prompt**: Instrucciones a nivel de sistema para guiar el comportamiento del modelo.
