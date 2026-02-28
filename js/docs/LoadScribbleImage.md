# Load Image & Scribble Editor

A dedicated image loader with a fully integrated HTML5 canvas built natively for ComfyUI. Use this node to intuitively draw arrows, add text, or freehand sketches directly over your reference images.

## Features
- **Integrated Scribble Canvas**: Click "Abrir Scribble Editor" to draw on top of your loaded image.
- **1:1 Resolution Scaling**: Your scribble data accurately inherits the native resolution of the base image regardless of screen dimension.
- **Brush & Asset Tools**: Easily switch colors, brush sizes, and embed icons/graphics on transparent layers perfectly sized for AI ingestion.

## Parameters
- **image**: The target image loaded from your input directory to display on the canvas.
- **scribble_data**: *(Hidden)* The base64 output data representing your scribble markup. Updates automatically when you save changes in the editor.

## Output
- **Image**: Discharges the original base image untouched (as a tensor variable).
- **scribble_mask**: Discharges the newly generated drawing as a separate RGBA (transparent background) tensor, properly formatted to feed directly into the **scribble** input of the Gemini Nano Banana Pro node.

## Usage Guide
1. Place a `LoadScribbleImage` node in your workspace.
2. Select your desired base image from the dropdown list.
3. Click the red **Abrir Scribble Editor** button attached to the node's properties.
4. Draw any desired markings or annotations, then hit **Save & Close** (Guardar y Cerrar).
5. Output `Image` into the `images` slot of GeminiNanoBananaPro.
6. Output `scribble_mask` into the `scribble` slot of GeminiNanoBananaPro to exert precision localized control.
