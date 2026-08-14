import base64
import importlib.util
import io
import json
import os
import pickle
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
from PIL import Image, ImageOps

# PyTorch 2.6+ defaults torch.load(weights_only=True), which blocks the
# official WiLoR-mini YOLO/MANO checkpoints (trusted HF weights).
_orig_torch_load = torch.load


def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
PIPE = None
MAX_SIDE = 1280


def patch_numpy_aliases():
    # Old chumpy does `from numpy import int, float, bool` (removed in NumPy 1.24+).
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


patch_numpy_aliases()


def pip_install(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def chumpy_imports():
    patch_numpy_aliases()
    try:
        import chumpy  # noqa: F401
        return True
    except Exception:
        return False


def ensure_chumpy():
    if chumpy_imports():
        return
    # Prefer a maintained fork; PyPI chumpy still uses removed NumPy aliases.
    for spec in (
        ["--no-build-isolation", "--force-reinstall", "git+https://github.com/uyoung-jeong/chumpy.git"],
        ["--no-build-isolation", "chumpy"],
    ):
        try:
            pip_install(*spec)
            if chumpy_imports():
                return
        except Exception:
            continue
    raise ModuleNotFoundError("No module named 'chumpy'")


def ensure_wilor():
    ensure_chumpy()
    if importlib.util.find_spec("wilor_mini") is None:
        pip_install("--no-deps", "git+https://github.com/warmshao/WiLoR-mini")
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )

    return WiLorHandPose3dEstimationPipeline


def get_pipe():
    global PIPE
    if PIPE is None:
        PIPE = ensure_wilor()(device=DEVICE, dtype=DTYPE, verbose=False)
    return PIPE


def load_mano_faces():
    mano = getattr(get_pipe().wilor_model, "mano", None)
    for attr in ("faces", "faces_tensor", "f"):
        raw = getattr(mano, attr, None)
        if raw is None:
            continue
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu()
        return np.asarray(raw).astype(int).tolist()
    try:
        root = os.path.dirname(importlib.util.find_spec("wilor_mini").origin)
        path = os.path.join(root, "pretrained_models", "MANO_RIGHT.pkl")
        with open(path, "rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        return np.asarray(data["f"]).astype(int).tolist()
    except Exception:
        return None


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (list, tuple, dict, str)):
        return len(value) == 0
    return False


def first_present(*values):
    for value in values:
        if not is_missing(value):
            return value
    return None


def looks_like_image(value):
    if value is None or isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim >= 2 and value.size > 16
    if isinstance(value, str):
        text = value.strip()
        return (
            text.startswith("data:image")
            or text.startswith("/")
            or os.path.exists(text)
            or len(text) > 200
        )
    return True


def pick_image(*values):
    for value in values:
        if looks_like_image(value):
            return value
    return None


def decode_image(payload):
    if payload is None:
        raise ValueError("No image")
    if isinstance(payload, dict):
        payload = payload.get("path") or payload.get("name") or payload.get("url") or payload.get("image")
        if payload is None:
            raise ValueError("No image")
    if isinstance(payload, np.ndarray):
        img = payload
    elif isinstance(payload, Image.Image):
        img = np.asarray(payload.convert("RGB"))
    elif isinstance(payload, (bytes, bytearray)):
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(payload)))
        img = np.asarray(pil.convert("RGB"))
    elif isinstance(payload, str):
        text = payload.strip()
        if text.startswith("data:"):
            raw = text.split(",", 1)[1]
            pil = ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(raw))))
        elif os.path.exists(text) or text.startswith("/"):
            pil = ImageOps.exif_transpose(Image.open(text))
        else:
            pil = ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(text))))
        img = np.asarray(pil.convert("RGB"))
    else:
        raise ValueError(f"Formato de imagen no soportado: {type(payload).__name__}")

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


def as_right(value):
    if is_missing(value):
        return True
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    elif isinstance(value, (list, tuple)):
        value = value[0]
    try:
        return float(value) >= 0.5
    except (TypeError, ValueError):
        return bool(value)


def parse_right_hint(value):
    if is_missing(value) or value == "":
        return None
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"left", "izquierda", "l", "false", "0"}:
            return False
        if text in {"right", "derecha", "r", "true", "1"}:
            return True
        return None
    return bool(value)


def anatomy_is_right(points_2d, points_3d=None):
    thumb, pinky = points_2d[4], points_2d[17]
    if abs(float(thumb[0]) - float(pinky[0])) < 8:
        return None
    seeing_back = True
    if not is_missing(points_3d) and len(points_3d) > 17:
        normal = np.cross(points_3d[5] - points_3d[0], points_3d[17] - points_3d[0])
        seeing_back = float(normal[2]) < 0
    thumb_left = float(thumb[0]) < float(pinky[0])
    if seeing_back:
        return not thumb_left
    return thumb_left


def fit_to_bbox(points_xy, bbox, pad=0.12):
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(x2 - x1, 1.0) * (1.0 + pad), max(y2 - y1, 1.0) * (1.0 + pad)
    lo = points_xy.min(axis=0)
    hi = points_xy.max(axis=0)
    sw, sh = max(float(hi[0] - lo[0]), 1e-6), max(float(hi[1] - lo[1]), 1e-6)
    scale = min(bw / sw, bh / sh)
    src = (lo + hi) / 2.0
    return (points_xy - src) * scale + np.array([cx, cy], dtype=np.float64)


def mostly_inside(points_xy, bbox):
    if is_missing(bbox) or len(bbox) < 4:
        return True
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    pad_x, pad_y = (x2 - x1) * 0.3, (y2 - y1) * 0.3
    inside = sum(
        1
        for point in points_xy
        if x1 - pad_x <= point[0] <= x2 + pad_x and y1 - pad_y <= point[1] <= y2 + pad_y
    )
    return inside >= 0.55 * len(points_xy)


def project_with_cam(points_3d, cam, width, height):
    cam = np.asarray(cam).reshape(-1)
    if cam.size < 3:
        return None
    scale, tx, ty = float(cam[0]), float(cam[1]), float(cam[2])
    xy = points_3d[:, :2] * scale + np.array([tx, ty], dtype=np.float64)
    span = float(np.nanmax(np.abs(xy))) if xy.size else 0
    if span < 3:
        xy = (xy + 1.0) * 0.5 * np.array([width, height], dtype=np.float64)
    return xy


def occlusion_flags(points_3d):
    palm_ids = [0, 5, 9, 13, 17]
    palm_z = float(np.mean(points_3d[palm_ids, 2]))
    palm_c = points_3d[palm_ids].mean(axis=0)
    normal = np.cross(points_3d[5] - points_3d[0], points_3d[17] - points_3d[0])
    if normal[2] > 0:
        normal = -normal
    flags = []
    for index, point in enumerate(points_3d):
        if index in palm_ids:
            flags.append(False)
            continue
        behind_depth = float(point[2]) > palm_z + 0.008
        behind_plane = float(np.dot(point - palm_c, normal)) < -0.004
        flags.append(bool(behind_depth or behind_plane))
    return flags


def infer(image_rgb, is_right_hint=None):
    img = decode_image(image_rgb)
    bgr = img[:, :, ::-1].copy()
    height, width = bgr.shape[:2]
    pipe = get_pipe()
    raw_first = pipe.predict(bgr, hand_conf=0.15)
    first = [] if raw_first is None else list(raw_first)
    hint = parse_right_hint(is_right_hint)
    bboxes = []
    rights = []
    for out in first:
        bbox = out.get("hand_bbox")
        preds = out.get("wilor_preds") or {}
        keypoints_2d = first_present(preds.get("pred_keypoints_2d"))
        keypoints_3d = first_present(preds.get("pred_keypoints_3d"))
        yolo_right = as_right(out.get("is_right", 1))
        chosen = yolo_right
        if hint is not None:
            chosen = hint
        elif not is_missing(keypoints_2d) and not is_missing(keypoints_3d):
            guessed = anatomy_is_right(as_points(keypoints_2d)[:, :2], as_points(keypoints_3d))
            if guessed is not None:
                chosen = guessed
        bboxes.append(bbox)
        rights.append(1.0 if chosen else 0.0)

    if first and any(as_right(out.get("is_right", 1)) != bool(rights[i]) for i, out in enumerate(first)):
        outputs = pipe.predict_with_bboxes(
            bgr,
            np.asarray(bboxes, dtype=np.float32),
            rights,
        )
    else:
        outputs = first

    hands = []
    for index, out in enumerate(outputs):
        preds = out.get("wilor_preds") or {}
        keypoints_2d = first_present(preds.get("pred_keypoints_2d"))
        keypoints_3d = first_present(preds.get("pred_keypoints_3d"))
        if is_missing(keypoints_2d) or is_missing(keypoints_3d):
            continue
        points_3d = as_points(keypoints_3d)
        count = min(21, len(points_3d))
        if count < 21:
            continue
        bbox = out.get("hand_bbox")
        frame_box = bbox if not is_missing(bbox) and len(bbox) >= 4 else [0, 0, width, height]
        kp2d = as_points(keypoints_2d)[:, :2]
        cam = first_present(preds.get("pred_cam"), preds.get("cam"))
        projected = project_with_cam(points_3d[:count], cam, width, height) if not is_missing(cam) else None
        if mostly_inside(kp2d[:count], frame_box):
            points_2d = kp2d[:count].astype(np.float64)
        elif projected is not None and mostly_inside(projected, frame_box):
            points_2d = projected
        else:
            points_2d = points_3d[:count, :2].copy()
            if not is_missing(bbox) and len(bbox) >= 4:
                points_2d = fit_to_bbox(points_2d, bbox, pad=0.08)
        hidden = occlusion_flags(points_3d[:count])
        is_right = as_right(rights[index] if index < len(rights) else out.get("is_right", 1))
        hand = {
            "hand": index + 1,
            "is_right": is_right,
            "bbox": [float(v) for v in bbox[:4]] if not is_missing(bbox) else None,
            "landmarks": [
                {
                    "id": joint,
                    "x": float(points_2d[joint, 0] / width),
                    "y": float(points_2d[joint, 1] / height),
                    "z": float(points_3d[joint, 2]),
                    "X": float(points_3d[joint, 0]),
                    "Y": float(points_3d[joint, 1]),
                    "Z": float(points_3d[joint, 2]),
                    "occluded": hidden[joint],
                }
                for joint in range(count)
            ],
        }
        verts = first_present(preds.get("pred_vertices"))
        if not is_missing(verts):
            points_v = as_points(verts)
            hand["vertices"] = [
                [float(row[0]), float(row[1]), float(row[2])] for row in points_v
            ]
        hands.append(hand)
    faces = load_mano_faces()
    payload = {
        "hands": hands,
        "faces": faces,
        "engine": "wilor-mini",
        "device": str(DEVICE),
        "size": [width, height],
    }
    if not hands:
        payload["error"] = "WiLoR no encontró una mano en la foto."
    return payload


@spaces.GPU(duration=90)
def predict(image=None, is_right_hint=None, *args, **kwargs):
    try:
        image = pick_image(image, is_right_hint, *args, *kwargs.values())
        hint = is_right_hint if not looks_like_image(is_right_hint) else kwargs.get("is_right")
        return json.dumps(infer(image, is_right_hint=hint), ensure_ascii=False)
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
            raw = predict(
                pick_image(body.get("image"), body.get("data"), body.get("file")),
                body.get("is_right"),
            )
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
    inputs=gr.Image(type="filepath", label="Foto de la mano"),
    outputs=gr.Textbox(label="21 landmarks (JSON)", lines=18),
    title="Hand WiLoR",
    description="Backend de WiLoR-mini para Hand Landmark Studio. Pegá una foto y obtené 21 puntos 2D/3D.",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.queue().launch(share=False, server_name="0.0.0.0", server_port=7860)
