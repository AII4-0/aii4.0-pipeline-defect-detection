import cv2
import numpy as np


def _get_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Remove the stripes
    kernel = np.ones((7, 7), np.uint8)
    dilated_image = cv2.dilate(gray, kernel, iterations=1)
    filled_stripes = cv2.erode(dilated_image, kernel, iterations=1)
    # Convert to binary image
    thresh = cv2.threshold(filled_stripes, 50, 255, cv2.THRESH_BINARY)[1]
    return thresh


def _align_img_rot(binary_mask, img):
    edges = cv2.Canny(binary_mask, 100, 255)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    # Get the line with the highest vote
    max_vote_line = max(lines, key=lambda line: line[0][0])

    # Extract the angle of the line
    _, theta = max_vote_line[0]
    rotation_angle = np.degrees(theta)
    # Rotate the image
    rows, cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
    rotated_image = cv2.warpAffine(img, rotation_matrix, (cols, rows))
    return rotated_image


def _align_img_pos(binary_mask, img, t_x=True, t_y=True):
    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        assert False, "No contours found"

    minx = min([cnt[:, 0, 0].min() for cnt in contours])
    maxx = max([cnt[:, 0, 0].max() for cnt in contours])
    miny = min([cnt[:, 0, 1].min() for cnt in contours])
    maxy = max([cnt[:, 0, 1].max() for cnt in contours])

    # create rectangle from min and max points
    rect_bb = np.array(
        [
            [[minx, miny]],
            [[maxx, miny]],
            [[maxx, maxy]],
            [[minx, maxy]],
        ]
    )

    # Get the center of the cross
    M = cv2.moments(rect_bb)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    # Get the center of the image
    rows, cols = img.shape[:2]
    tx = cols // 2 - cx if t_x else 0
    ty = rows // 2 - cy if t_y else 0

    # Perform the translation
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    centered_image = cv2.warpAffine(img, translation_matrix, (cols, rows))
    return centered_image


def align_img(img):
    mask = _get_mask(img)
    img_pos = _align_img_pos(mask, img)
    mask_rot = _get_mask(img_pos)
    img_aligned = _align_img_rot(mask_rot, img_pos)

    rows, cols = img_aligned.shape[:2]
    x_crop = (cols - rows) // 2
    return img_aligned[:, x_crop : cols - x_crop]


def get_img_crops(img):
    # img is a cross with a black background
    # We now extract 4 images from it, each one representing a side of the cross
    img = img.copy()
    rows, cols = img.shape[:2]
    crop_top = img[: rows // 3, rows // 3 : 2 * rows // 3]
    crop_bottom = img[-rows // 3 :, cols // 3 : 2 * cols // 3]
    crop_left = img[rows // 3 : 2 * rows // 3, : cols // 3]
    crop_right = img[rows // 3 : 2 * rows // 3, -cols // 3 :]

    # rotate the images to have the same orientation
    cv2.rotate(crop_top, cv2.ROTATE_90_CLOCKWISE, crop_top)
    cv2.rotate(crop_bottom, cv2.ROTATE_90_COUNTERCLOCKWISE, crop_bottom)
    cv2.rotate(crop_left, cv2.ROTATE_180, crop_left)
    new_crops = []
    left_crop_percent = 40
    y_crop_percent = 30
    for crop in [crop_top, crop_bottom, crop_left, crop_right]:
        mask = _get_mask(crop)
        crop = _align_img_pos(mask, crop, t_x=False)

        width = crop.shape[1]
        height = crop.shape[0]
        left_crop = width * left_crop_percent // 100
        y_crop = height * y_crop_percent // 100
        # crop y axis so that the image is a square
        # y_crop = (height - width) // 2
        new_crops.append(crop[y_crop:-y_crop, left_crop:])

    return new_crops


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Remove the stripes
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(gray, kernel, iterations=1)
    filled_stripes = cv2.erode(dilated_image, kernel, iterations=1)
    # Convert to binary image
    thresh = cv2.threshold(filled_stripes, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.resize(thresh, (64, 64))
    return thresh


def preprocess_crop(crop):
    crop = preprocess(crop)
    crop = crop.astype(np.float32) / 255
    return crop.reshape((1, 1, *crop.shape))
