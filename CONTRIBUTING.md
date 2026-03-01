# Guía de Contribución para ComfyUI-NanoBananaPro

¡Primero que nada, gracias por tomarte el tiempo para contribuir! ❤️

Todas las contribuciones son bienvenidas, desde reportar errores hasta proponer nuevas características y mejorar la documentación. Este documento contiene pautas para ayudarte a contribuir en este proyecto.

## 🐛 Reportar Errores (Bugs)

Si encuentras un error o el nodo no funciona como esperas, por favor abre un nuevo "Issue" (problema) en el repositorio y asegúrate de incluir:

* Una descripción clara y concisa del problema.
* Los pasos exactos para reproducir el error.
* El archivo JSON de tu flujo de trabajo (workflow) de ComfyUI donde ocurre el error (si aplica).
* Logs de la consola o mensajes de error de ComfyUI.
* Tu sistema operativo y versión de Python.

## 💡 Sugerir Mejoras o Funcionalidades

Las ideas para mejorar el nodo son muy valiosas. Si tienes una sugerencia, abre un "Issue" explicando:

* El propósito de tu idea y por qué sería útil para otros usuarios.
* Ejemplos de cómo funcionaría o qué problema resuelve.
* Si es posible, URLs o referencias a APIs que podrían ser útiles para la implementación.

## 💻 Entorno de Desarrollo

Si quieres aportar código, debes configurar tu entorno de desarrollo local:

1. Asegúrate de tener ComfyUI instalado y funcionando.
2. Ve a la carpeta `custom_nodes` de tu instalación de ComfyUI:
   ```bash
   cd ComfyUI/custom_nodes
   ```
3. Clona (Fork) este repositorio en tu cuenta de GitHub y luego clónalo a tu máquina local:
   ```bash
   git clone https://github.com/TU_USUARIO/ComfyUI-NanoBananaPro.git
   ```
4. Navega dentro de la carpeta e instala cualquier dependencia requerida (si existe un `requirements.txt`):
   ```bash
   cd ComfyUI-NanoBananaPro
   pip install -r requirements.txt
   ```
5. Reinicia ComfyUI para que cargue los nodos modificados.

## 🔀 Proceso de Pull Requests (PR)

Para enviar tus cambios, sigue estos pasos:

1. **Crea una rama (branch)** para tu funcionalidad o corrección:
   ```bash
   git checkout -b feature/nombre-de-tu-funcionalidad
   ```
   *Usa prefijos como `feature/`, `fix/`, o `docs/` para mantener el orden.*

2. **Haz tus cambios** e intenta mantener el código limpio y documentado.
3. **Prueba tus cambios** localmente en ComfyUI asegurándote de que no rompen los flujos de trabajo existentes.
4. **Haz commit** de tus cambios con mensajes claros:
   ```bash
   git commit -m "Añade: soporte para modelos Pro en la generación"
   ```
5. **Sube tu rama** (Push) a tu repositorio (Fork):
   ```bash
   git push origin feature/nombre-de-tu-funcionalidad
   ```
6. **Abre un Pull Request** en el repositorio original describiendo claramente qué cambios hiciste y por qué.

## 🎨 Guía de Estilo de Código

* El código Python debe seguir [PEP 8](https://peps.python.org/pep-0008/) tanto como sea posible.
* Mantén las clases y métodos bien comentados, especialmente la estructura `INPUT_TYPES` y `RETURN_TYPES` si modificas la interfaz gráfica del nodo.
* Si agregas o cambias requerimientos, asegúrate de actualizar el archivo `requirements.txt` o `pyproject.toml`.

¡Gracias por ayudar a hacer de este proyecto algo mejor!🚀
