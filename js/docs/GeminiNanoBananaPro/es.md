# Nano Banana Pro (Gemini Direct)

Este nodo integra los modelos **Gemini** y **Vertex AI (Imagen 3)** de Google de forma directa en ComfyUI, permitiendo la generación de imágenes de alta velocidad y edición avanzada.

## Características
- Desbloquea **Gemini 3 Pro** y **Gemini 3.1 Flash (Nano Banana 2)** para la generación de imágenes.
- Soporta **Imagen 3** para tareas de edición avanzadas como Inpainting, Outpainting y Cambio de Fondo.
- Soporte multilinea nativo para prompts y composición de imágenes con altos límites de API.

## Parámetros

### Requeridos
- **prompt**: Descripción en texto de la imagen a generar.
- **model**: Modelo de IA a utilizar:
  - `gemini-3.1-flash-image-preview` / `gemini-3-pro-image-preview`: Para la generación rápida de imágenes (requiere API Key de Google AI Studio).
  - `imagen-*`: Operaciones avanzadas de edición (requiere JSON Key de Vertex AI).
- **operation**: Acción a realizar:
  - `GENERATE`: Generación estándar de Texto a Imagen o Imagen a Imagen.
  - `INPAINT_INSERTION` / `INPAINT_REMOVAL` / `OUTPAINT` / `BACKGROUND_SWAP`: Edición de imagen (Requiere Imagen + Vertex AI).

### Opcionales
- **api_key**: Clave API de Google AI Studio (para modelos Gemini). Alternativamente, puede ser la ruta local a un archivo .txt.
- **service_account_json**: Ruta absoluta a un archivo de clave JSON de Service Account de Vertex AI en Google Cloud (para Imagen 3 y Edición).
- **images**: Imagen base para operaciones como Img2Img o edición con Imagen.
- **mask**: Imagen de máscara para inpainting (Blanco = Editar, Negro = Conservar).
- **scribble**: Imagen de boceto o garabato para inpainting controlado. Se puede crear de forma nativa con el nodo *Load Image & Scribble Editor*.
- **seed**: Semilla inicial para la generación de la imagen.
- **aspect_ratio**: Relación de aspecto del resultado a generar (1:1, 16:9, etc.).
- **resolution**: Nivel de resolución de la imagen generada (1K, 2K, 4K).
- **response_modalities**: Modalidad de la respuesta, ej: sólo la imagen (`IMAGE`) u obtener los razonamientos de la inteligencia artificial (`IMAGE+TEXT`).
- **system_prompt**: Instrucciones a nivel del sistema base para darle un contexto específico o general a la IA.

## Modo de Uso
1. **Generación con Gemini**: Ingresa una `api_key`, selecciona el modelo Gemini de preferencia y usa la operación `GENERATE`.
2. **Edición con Imagen 3**: Ingresa la ruta al `service_account_json` de Vertex AI, conecta un nodo de `images` y, de ser necesario, `mask` o `scribble`. Selecciona el modelo `imagen-3.0-capability-001` junto con la operación deseada (ej. `INPAINT_INSERTION`).
