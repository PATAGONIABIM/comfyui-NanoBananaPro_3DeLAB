# Load Image & Scribble Editor

Un cargador de imágenes personalizado que incluye un Canvas HTML5 totalmente integrado dentro de ComfyUI. Usa este nodo para trazar flechas, añadir texto o realizar bocetos a mano alzada directamente sobre tus imágenes de referencia. 

## Características
- **Lienzo Gráfico Integrado**: Presiona "Abrir Scribble Editor" para pintar encima de la imagen base que elijas.
- **Escalado en Resolución 1:1**: Los trazos heredan con asombrosa precisión la resolución nativa de tu imagen de fondo, incluso si cambias de pantalla o acercas la vista.
- **Herramientas de Pincel y Diseño**: Cambia el grosor, el color y la goma de borrar para generar capas intermedias listas para enviar al modelo de IA.

## Parámetros
- **image**: Imagen de fondo cargada desde tu directorio local (`input/`) que servirá como mapa de referencia.
- **scribble_data**: *(Oculto)* Datos de codificación base64 que guardan el dibujo resultante. Se auto-actualiza al guardar en el editor gráfico.

## Salidas (Outputs)
- **Image**: Devuelve el tensor con la imagen base inalterada y lista para uso posterior.
- **scribble_mask**: Devuelve el nuevo dibujo aislado en una imagen RGBA (fondo transparente) y correctamente formateada para enviarse por el pin de **scribble** del nodo de Nano Banana Pro.

## Modo de Uso
1. Coloca un nodo `LoadScribbleImage` en tu espacio de trabajo.
2. Selecciona la imagen base en el menú desplegable.
3. Presiona el botón rojo **Abrir Scribble Editor**.
4. Dibuja las marcas o bocetos deseados, luego guárdalo eligiendo **Guardar y Cerrar**.
5. Conecta la el pin de salida `Image` a la entrada `images` del nodo de generación (GeminiNanoBananaPro).
6. Conecta el pin de salida `scribble_mask` a la entrada `scribble` del nodo de generación para otorgarle a la inteligencia artificial un control preciso local.
