import bentoml
import cv2
import numpy as np
from bentoml.io import JSON, Image, Multipart
from PIL import Image as PIL_Image
from pydantic import BaseModel

from src.utils.preprocessing import align_img, get_img_crops, preprocess_crop

runner = bentoml.onnx.get("onnx_defect_detection_api:latest").to_runner()

svc = bentoml.Service("onnx_defect_detection_api", runners=[runner])


class OptsDict(BaseModel):
    threshold: float = 0.5


class OutputPredsDict(BaseModel):
    top: float
    bottom: float
    left: float
    right: float


class OutputDict(BaseModel):
    has_defect: bool
    preds: OutputPredsDict


@svc.api(
    input=Multipart(img=Image(), opts=JSON(pydantic_model=OptsDict)),
    output=JSON(pydantic_model=OutputDict),
)
async def predict_one(img: PIL_Image, opts: OptsDict) -> OutputDict:
    # convert to numpy array
    img = np.array(img)
    img_aligned = align_img(img)
    crops = get_img_crops(img_aligned)
    batch = np.concatenate([preprocess_crop(crop) for crop in crops], axis=0)
    preds = await runner.run.async_run(batch)

    has_defect = False
    defect_threshold = opts.threshold
    pred_dict = {}
    for i, pos in enumerate(["top", "bottom", "left", "right"]):
        pred = preds[i][0]
        pred_dict[pos] = pred
        if pred >= defect_threshold:
            has_defect = True

    return {"has_defect": has_defect, "preds": {**pred_dict}}


@svc.api(
    input=Multipart(img=Image(), opts=JSON(pydantic_model=OptsDict)),
    output=Image(mime_type="image/png"),
)
async def preview_preds_one(img: PIL_Image, opts: OptsDict) -> np.ndarray:
    img = np.array(img)
    img_aligned = align_img(img)
    crops = get_img_crops(img_aligned)
    batch = np.concatenate([preprocess_crop(crop) for crop in crops], axis=0)
    preds = await runner.run.async_run(batch)

    img_aligned_labed = img_aligned.copy()
    defect_threshold = opts.threshold
    text_pad = 10

    for i, pos in enumerate(["top", "bottom", "left", "right"]):
        pred = preds[i][0]
        if pred >= defect_threshold:  # defect
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
    return img_aligned_labed


@svc.api(input=Image(), output=Image(mime_type="image/png"))
async def align_one(img: PIL_Image) -> np.ndarray:
    # convert to numpy array
    img = np.array(img)
    return align_img(img)
