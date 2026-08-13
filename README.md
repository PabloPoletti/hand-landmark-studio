# Hand Landmark Studio

Sitio estático para pegar una foto de una mano (`Ctrl+V`) y ver los **21 landmarks** de [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) sobre la imagen y en tres vistas 3D: **frente**, **3/4** y **lateral**.

El esquema es el mismo que Blueprint / MediaPipe: `0` muñeca, pulgar `1–4`, índice `5–8`, medio `9–12`, anular `13–16`, meñique `17–20`.

## Por qué no corre WiLoR-mini acá

[WiLoR-mini](https://github.com/PabloPoletti/WiLoR-mini) estima pose 3D con PyTorch, YOLO, MANO y un checkpoint grande. Eso no se puede publicar en GitHub Pages.

Este sitio usa el landmarker oficial en el navegador: no hay backend, no hay API key y no se sube la foto a ningún servidor.

## Uso

1. Abrí el sitio.
2. Esperá a que diga **Modelo listo**.
3. Pegá una captura (`Ctrl+V`), soltá un archivo o hacé clic para elegir uno.
4. Si hay dos manos, cambiá de track con los chips.
5. **Copiar JSON 21 pts** exporta coordenadas normalizadas `0–1`, compatibles con el import de la extensión Blueprint.

Las vistas 3D se pueden rotar con el mouse.

## Desarrollo local

Cualquier servidor estático alcanza (los módulos ES no abren bien con `file://`):

```bash
npx --yes serve .
```

## GitHub Pages

El sitio se sirve desde la rama `main`, carpeta raíz.

## Backend WiLoR (opcional, Hugging Face)

MediaPipe corre solo. El botón **Refinar con WiLoR** llama a un [Space](https://huggingface.co/docs/hub/spaces-overview) gratis.

1. Creá una cuenta en [huggingface.co](https://huggingface.co/join) (gratis).
2. En [New Space](https://huggingface.co/new-space) elegí Gradio, Python 3.10, hardware **CPU basic**.
3. Subí los archivos de `space/` (`app.py`, `requirements.txt`, `README.md`).
4. Cuando el Space esté `Running`, copiá la URL `https://xxxxx.hf.space` en `js/config.js`.

La primera inferencia tarda porque descarga los pesos. Después, cada foto puede llevar 10–40 s en CPU.
