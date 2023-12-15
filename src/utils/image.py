import cv2
import numpy as np
import torch


def label_pred_images(
    x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor, border_size: int = 2
) -> np.ndarray:
    """Label images with red or green border depending on whether the prediction was correct"""
    arr_batch = np.ndarray((x.shape[0], 3, x.shape[-2], x.shape[-1]))
    for i in range(len(x)):
        target_class = y[i]
        # y_hat[i] is a tensor of shape (1,) with the predicted class
        pred_class = y_hat[i].cpu().detach().round()
        if target_class == pred_class:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        img = x[i]
        img_arr = (
            (img.permute(1, 2, 0)).cpu().detach().numpy().astype("uint8").squeeze()
        )
        # Remove 'border_size' pixels from each side
        img_arr = img_arr[border_size:-border_size, border_size:-border_size]
        # Convert to RGB if necessary
        if len(img_arr.shape) == 2:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        # Add border
        img_arr = cv2.copyMakeBorder(
            img_arr,
            border_size,
            border_size,
            border_size,
            border_size,
            cv2.BORDER_CONSTANT,
            value=color,
        )
        img_arr = img_arr.transpose(2, 0, 1)
        arr_batch[i] = img_arr

    return arr_batch
