import base64
import importlib.util
import io
import json
import subprocess
import sys
import traceback

import numpy as np
import spaces
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
PIPE = None
MAX_SIDE = 1280

PAGE = """<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hand WiLoR</title>
  <style>
    body { font-family: sans-serif; background: #0f1419; color: #e7eef7; margin: 24px; }
    textarea { width: 100%; min-height: 280px; background: #151b22; color: #e7eef7; border: 1px solid #2a3440; }
    button { background: #f97316; color: #111; border: 0; padding: 10px 16px; font-weight: 700; cursor: pointer; }
    .err { color: #fca5a5; }
  </style>
</head>
<body>
  <h1>Hand WiLoR</h1>
  <p>Backend de WiLoR-mini para Hand Landmark Studio. Pegá o elegí una foto.</p>
  <input id="file" type="file" accept="image/*" />
  <button id="go">Detectar</button>
  <p id="status"></p>
  <textarea id="out" readonly></textarea>
  <script>
    const file = document.getElementById("file");
    const out = document.getElementById("out");
    const status = document.getElementById("status");
    async function toDataUrl(f) {
      return await new Promise((resolve, reject) => {
        const r = new FileReader();
        r.onload = () => resolve(r.result);
        r.onerror = reject;
        r.readAsDataURL(f);
      });
    }
    async function run(f) {
      status.textContent = "Procesando… la primera vez puede tardar 1–2 min";
      out.value = "";
      const res = await fetch("/wilor", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: await toDataUrl(f) }),
      });
      const data = await res.json();
      out.value = JSON.stringify(data, null, 2);
      status.textContent = data.error
        ? data.error
        : ((data.hands || []).length + " mano(s)");
      status.className = data.error ? "err" : "";
    }
    document.getElementById("go").onclick = () => file.files[0] && run(file.files[0]);
    document.addEventListener("paste", (e) => {
      const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
      if (item) run(item.getAsFile());
    });
  </script>
</body>
</html>
"""


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


@spaces.GPU(duration=90)
def run_wilor(image_rgb):
    img = decode_image(image_rgb)
    # WiLoR-mini / YOLO were used with OpenCV BGR in the official examples.
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


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


@app.get("/health")
def health():
    return {"ok": True, "device": str(DEVICE)}


@app.post("/wilor")
async def wilor_api(request: Request):
    try:
        body = await request.json()
        return JSONResponse(run_wilor(body.get("image")))
    except Exception as exc:
        return JSONResponse(
            {
                "hands": [],
                "error": str(exc),
                "trace": traceback.format_exc()[-2000:],
            },
            status_code=200,
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
