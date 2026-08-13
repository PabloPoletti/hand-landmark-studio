import importlib.util
import json
import subprocess
import sys

import numpy as np
import torch
import gradio as gr
import spaces
from gradio_client import utils as gradio_schema

# Gradio 4.44 crashes when additionalProperties is a bool (True/False),
# not a schema object. Patch before building the UI.
_orig_get_type = gradio_schema.get_type


def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)


gradio_schema.get_type = _safe_get_type

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
PIPE = None


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
        pipeline_cls = ensure_wilor()
        PIPE = pipeline_cls(device=DEVICE, dtype=DTYPE, verbose=False)
    return PIPE


def pack(payload):
    return json.dumps(payload)


@spaces.GPU(duration=90)
def predict(image):
    if image is None:
        return pack({"hands": [], "error": "No image"})

    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    height, width = img.shape[:2]

    outputs = get_pipe().predict(img)
    hands = []
    for index, out in enumerate(outputs):
        preds = out.get("wilor_preds") or {}
        keypoints_2d = preds.get("pred_keypoints_2d")
        keypoints_3d = preds.get("pred_keypoints_3d")
        if keypoints_2d is None or keypoints_3d is None:
            continue
        points_2d = np.asarray(keypoints_2d[0])
        points_3d = np.asarray(keypoints_3d[0])
        count = min(21, len(points_2d), len(points_3d))
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
    return pack({"hands": hands, "engine": "wilor-mini", "device": str(DEVICE)})


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
