import base64
import importlib.util
import io
import json
import subprocess
import sys
import traceback

import gradio as gr
import numpy as np
import spaces
import torch
from fastapi import Request
from fastapi.responses import JSONResponse
from gradio.routes import App
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
PIPE = None
MAX_SIDE = 1280


def ensure_wilor():
    if importlib.util.find_spec("wilor_mini") is None:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "git+https://github.com/warmshao/WiLoR-mini",
            ]
        )
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )

    return WiLorHandPose3dEstimationPipeline


def get_pipe():
    global PIPE
    if PIPE is None:
        PIPE = ensure_wilor()(device=DEVICE, dtype=DTYPE, verbose=False)
    return PIPE


def decode_image(payload):
    if payload is None:
        raise ValueError("No image")
    if isinstance(payload, np.ndarray):
        img = payload
    elif isinstance(payload, Image.Image):
        img = np.asarray(payload.convert("RGB"))
    elif isinstance(payload, str):
        raw = payload.split(",", 1)[1] if payload.startswith("data:") else payload
        img = np.asarray(Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB"))
    else:
        raise ValueError("Formato de imagen no soportado")

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    height, width = img.shape[:2]
    longest = max(height, width)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / longest
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        img = np.asarray(Image.fromarray(img).resize(new_size, Image.Resampling.BILINEAR))
    return img


def as_points(values):
    points = np.asarray(values)
    if points.ndim == 3:
        points = points[0]
    return points


def infer(image_rgb):
    img = decode_image(image_rgb)
    bgr = img[:, :, ::-1].copy()
    height, width = bgr.shape[:2]
    outputs = get_pipe().predict(bgr, hand_conf=0.15)
    hands = []
    for index, out in enumerate(outputs):
        preds = out.get("wilor_preds") or {}
        keypoints_2d = preds.get("pred_keypoints_2d")
        keypoints_3d = preds.get("pred_keypoints_3d")
        if keypoints_2d is None or keypoints_3d is None:
            continue
        points_2d = as_points(keypoints_2d)
        points_3d = as_points(keypoints_3d)
        count = min(21, len(points_2d), len(points_3d))
        if count < 21:
            continue
        hands.append(
            {
                "hand": index + 1,
                "is_right": bool(out.get("is_right", 1)),
                "landmarks": [
                    {
                        "id": joint,
                        "x": float(points_2d[joint, 0] / width),
                        "y": float(points_2d[joint, 1] / height),
                        "z": float(points_2d[joint, 2]) if points_2d.shape[1] > 2 else 0.0,
                        "X": float(points_3d[joint, 0]),
                        "Y": float(points_3d[joint, 1]),
                        "Z": float(points_3d[joint, 2]),
                    }
                    for joint in range(count)
                ],
            }
        )
    payload = {
        "hands": hands,
        "engine": "wilor-mini",
        "device": str(DEVICE),
        "size": [width, height],
    }
    if not hands:
        payload["error"] = "WiLoR no encontró una mano en la foto."
    return payload


@spaces.GPU(duration=90)
def predict(image):
    try:
        return json.dumps(infer(image), ensure_ascii=False)
    except Exception as exc:
        return json.dumps(
            {
                "hands": [],
                "error": str(exc),
                "trace": traceback.format_exc()[-2000:],
            },
            ensure_ascii=False,
        )


def static_api_info(self, *args, **kwargs):
    return {
        "named_endpoints": {
            "/predict": {
                "parameters": [
                    {
                        "label": "Foto de la mano",
                        "parameter_name": "image",
                        "parameter_has_default": False,
                        "type": {"type": "string"},
                        "python_type": {"type": "filepath", "description": ""},
                        "component": "Image",
                        "example_input": "",
                    }
                ],
                "returns": [
                    {
                        "label": "21 landmarks (JSON)",
                        "type": {"type": "string"},
                        "python_type": {"type": "str", "description": ""},
                        "component": "Textbox",
                    }
                ],
            }
        },
        "unnamed_endpoints": {},
    }


gr.blocks.Blocks.get_api_info = static_api_info

_orig_create_app = App.create_app


def create_app_with_wilor(blocks, *args, **kwargs):
    app = _orig_create_app(blocks, *args, **kwargs)

    @app.post("/wilor")
    async def wilor_api(request: Request):
        try:
            body = await request.json()
            raw = predict(body.get("image"))
            data = json.loads(raw) if isinstance(raw, str) else raw
            return JSONResponse(data)
        except Exception as exc:
            return JSONResponse(
                {
                    "hands": [],
                    "error": str(exc),
                    "trace": traceback.format_exc()[-2000:],
                }
            )

    @app.get("/health")
    def health():
        return {"ok": True, "device": str(DEVICE)}

    return app


App.create_app = staticmethod(create_app_with_wilor)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Foto de la mano"),
    outputs=gr.Textbox(label="21 landmarks (JSON)", lines=18),
    title="Hand WiLoR",
    description="Backend de WiLoR-mini para Hand Landmark Studio. Pegá una foto y obtené 21 puntos 2D/3D.",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.queue().launch(share=False, server_name="0.0.0.0", server_port=7860)
