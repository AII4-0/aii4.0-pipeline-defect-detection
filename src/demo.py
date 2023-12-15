import argparse
from functools import partial
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import onnxruntime
import yaml

from src.utils.preprocessing import align_img, get_img_crops, preprocess_crop


def predict(
    ort_session: onnxruntime.InferenceSession, img: np.ndarray, threshold: float = 0.5
):
    if img is None:
        return None, None
    img_aligned = align_img(img)
    img_aligned_labed = img_aligned.copy()
    pred_dict = {}
    text_pad = 10
    crops = get_img_crops(img_aligned)

    batch = np.concatenate([preprocess_crop(crop) for crop in crops], axis=0)
    preds = ort_session.run(None, {ort_session.get_inputs()[0].name: batch})[0]

    for i, pos in enumerate(["top", "bottom", "left", "right"]):
        pred = preds[i][0]
        pred_dict[pos.capitalize()] = pred
        if pred >= threshold:  # defect
            rows, cols = img_aligned_labed.shape[:2]
            if pos == "top":
                pt1 = (cols // 3, 0)
                pt2 = (2 * cols // 3, rows // 3)
            elif pos == "bottom":
                pt1 = (cols // 3, 2 * rows // 3)
                pt2 = (2 * cols // 3, rows)
            elif pos == "left":
                pt1 = (0, rows // 3)
                pt2 = (cols // 3, 2 * rows // 3)
            elif pos == "right":
                pt1 = (2 * cols // 3, rows // 3)
                pt2 = (cols, 2 * rows // 3)
            cv2.rectangle(
                img_aligned_labed,
                pt1,
                pt2,
                (255, 0, 0),
                5,
            )
            cv2.putText(
                img_aligned_labed,
                f"{pred:.2f}",
                (pt1[0] + text_pad, pt2[1] - text_pad),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

    return img_aligned_labed, pred_dict


def main(args: argparse.Namespace) -> None:
    base_dir = Path("data/datasets/serie02_plus_serie01_png")
    examples = [
        # With defects
        "30057.png",
        "30014_2.png",
        "30143.png",
        # Without defects
        "10004.png",
        "10012.png",
        "10035.png",
    ]
    params = yaml.safe_load(open("params.yaml"))
    model_path = Path(params["export"]["out_path"]) / "model.onnx"

    ort_session = onnxruntime.InferenceSession(model_path)

    gr.Interface(
        fn=partial(predict, ort_session),
        inputs=[
            gr.Image(type="numpy", label="Input image"),
            gr.Slider(0, 1, 0.5, step=0.01, label="Threshold"),
        ],
        outputs=[
            gr.Image(type="numpy", show_label=False),
            gr.Label(num_top_classes=4, label="Defect probabilities"),
        ],
        examples=[[str(base_dir / example)] for example in examples],
        title="AII4.0 SO3 Pipeline3 - Defect detection",
        description="Detect defects on top-down images of tools.",
        allow_flagging="never",
        analytics_enabled=False,
    ).launch(server_name="0.0.0.0", share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to create a publicly shareable link for the demo",
    )
    args = parser.parse_args()
    main(args)
