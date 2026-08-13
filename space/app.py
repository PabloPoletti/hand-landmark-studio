import importlib.util
import subprocess
import sys

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

import numpy as np
import torch
import gradio as gr
import spaces
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
    WiLorHandPose3dEstimationPipeline,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
PIPE = None


def get_pipe():
    global PIPE
    if PIPE is None:
        PIPE = WiLorHandPose3dEstimationPipeline(
            device=DEVICE, dtype=DTYPE, verbose=False
        )
    return PIPE


@spaces.GPU(duration=90)
def predict(image):
    if image is None:
        return {"hands": [], "error": "No image"}

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
    return {"hands": hands, "engine": "wilor-mini", "device": str(DEVICE)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Foto de la mano"),
    outputs=gr.JSON(label="21 landmarks"),
    title="Hand WiLoR",
    description="Backend de WiLoR-mini para Hand Landmark Studio. Pegá una foto y obtené 21 puntos 2D/3D.",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.queue().launch()
