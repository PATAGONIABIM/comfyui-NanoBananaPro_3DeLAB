# Gemini Veo 3.1 & 2.0 Video Generator

Este nodo proporciona funciones de generación y edición de video utilizando los modelos de última generación **Veo 3.1** y **Veo 2.0** de Google.

## Características
- **Texto-a-Video**: Generación de video estándar usando `veo-3.1-generate-preview` a través de tu API Key de Google AI Studio.
- **Imagen-a-Video (Reference Images)**: Genera videos basándote en hasta 3 imágenes de referencia utilizando `veo-2.0-generate-exp` vía Vertex AI (Requiere Service Account JSON y GCS Bucket).
- **Edición de Video**: Inserta objetos en videos (Inpaint), remuévelos o extiende clips existentes usando `veo-2.0-generate-preview` (Requiere configuración de Vertex AI).

## Parámetros

### Requeridos
- **prompt**: Prompt de texto que describe el video o la edición deseada.
- **model**: Selecciona el modelo de video a usar (`veo-3.1-generate-preview`, `veo-2.0-generate-exp`, etc.).
- **mode**: Operación a realizar (`standard`, `reference images`, `inpaint_insertion`, `inpaint_removal`, `extend_video`).
- **api_key**: Clave API de Google AI Studio (Requerida para Veo 3.1 estándar).
- **service_account_json**: Ruta absoluta al JSON de Service Account de Vertex AI (Requerido para Veo 2.0 y operaciones de edición).
- **gcs_bucket**: Nombre de tu bucket de Google Cloud Storage (Requerido para subir video/máscaras/imágenes durante operaciones de Vertex AI).

### Entradas Opcionales
- **image**: Imagen(es) de referencia para el modo `reference images` (puedes enviar un lote de hasta 3 imágenes).
- **video**: Video base para inpainting o extensión.
- **mask**: Imagen de máscara que define la región a editar en inpainting.

## Guía de Uso
1. **`txt2vid` estándar (Veo 3.1)**: Configura el modo en `standard`, usa `veo-3.1-generate-preview`, y provee tu `api_key`.
2. **Reference Images `img2vid` (Veo 2.0)**: Usa `veo-2.0-generate-exp`, cambia el modo a `reference images`, conecta hasta 3 imágenes en lote al input `image`, y provee tu `service_account_json` junto a un `gcs_bucket` válido.
3. **Inpainting de Video (Veo 2.0)**: Usa `veo-2.0-generate-preview`, establece el modo en `inpaint_insertion` o `inpaint_removal`. Conecta un `video` y su `mask`. Para procesarlo en la nube provee tu `service_account_json` y tu `gcs_bucket` para la carga automatizada de archivos.
