import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


def get_overlap_img(test_img_path: str, mask_img_path: str) -> np.ndarray:
    if isinstance(test_img_path, Path):
        test_img_path = str(test_img_path)
    if isinstance(mask_img_path, Path):
        mask_img_path = str(mask_img_path)

    test_img = cv2.imread(test_img_path)
    if type(mask_img_path) == str:
        mask = cv2.imread(mask_img_path)
    else:
        mask = mask_img_path

    # Convert mask to 3 channels
    if len(mask.shape) == 2:
        mask = np.stack([mask] * 3, axis=-1)
    mask = cv2.resize(mask, (test_img.shape[1], test_img.shape[0]),
                      interpolation=cv2.INTER_LINEAR)

    test_img = test_img.astype(np.uint8)
    mask = mask.astype(np.uint8)
    try:
        overlap = cv2.addWeighted(test_img, 0.5, mask, 0.5, 0)
    except Exception as e:
        print("test_img shape:", test_img.shape, "mask shape:", mask.shape)
        raise e
    return overlap


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255 / 255, 255 / 255, 255 / 255, 0.3])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
            for contour in contours
        ]
        mask_image = cv2.drawContours(mask_image,
                                      contours,
                                      -1, (1, 1, 1, 0.5),
                                      thickness=1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=300):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0],
               pos_points[:, 1],
               color='yellowgreen',
               marker='.',
               s=marker_size,
               edgecolor='white',
               linewidth=0)
    ax.scatter(neg_points[:, 0],
               neg_points[:, 1],
               color='red',
               marker='.',
               s=marker_size,
               edgecolor='white',
               linewidth=0)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0),
                      w,
                      h,
                      edgecolor='yellowgreen',
                      facecolor=(0, 0, 0, 0),
                      lw=2))


def show_boxes(boxes, ax):
    if boxes.ndim == 1:
        show_box(boxes, ax)
        return
    for box in boxes:
        show_box(box, ax)


def show_masks(img_path,
               masks,
               scores,
               point_coords=None,
               box_coords=None,
               input_labels=None,
               borders=True,
               mode='horizontal',
               save_path='show_mask.png'):

    masks = list(masks)
    n = len(masks)

    if mode == 'horizontal':
        image = Image.open(img_path)
        image = np.array(image.convert("RGB"))
        # single-row figure, one subplot per mask
        figsize = (5 * max(1, n), 5)
        fig, axes = plt.subplots(1, n, figsize=figsize)
        # ensure axes is iterable even for n==1
        if n == 1:
            axes = [axes]

        for i, (mask, score, ax) in enumerate(zip(masks, scores, axes)):
            ax.imshow(np.uint8(image))
            show_mask(mask, ax, random_color=False, borders=borders)
            if point_coords is not None:
                assert input_labels is not None
                show_points(point_coords, input_labels, ax)
            if box_coords is not None:
                show_boxes(box_coords, ax)
            if len(scores) > 1:
                ax.set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=14)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path)

    elif mode == 'singlemask+multiboxes':
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        combined_img = np.zeros((h, w), dtype=np.uint8)
        for i, mask in enumerate(masks):
            mbin = (mask > 0.5).astype(np.uint8) * 255
            combined_img = np.maximum(combined_img, mbin)
        overlap_img = get_overlap_img(img_path, combined_img)

        for box in box_coords:
            x0, y0 = box[0], box[1]
            w_box, h_box = box[2] - box[0], box[3] - box[1]
            overlap_img = cv2.rectangle(overlap_img, (x0, y0),
                                        (x0 + w_box, y0 + h_box), (0, 255, 0),
                                        2)
        cv2.imwrite(str(save_path), overlap_img)
    else:
        raise NotImplementedError


def get_inputs_from_mask(mask_path):
    mask_pil = Image.open(mask_path).convert("L")
    mask = np.array(mask_pil)
    th = 127 if mask.max() > 1 else 0.5

    bin_mask = (mask > th).astype(np.uint8)

    # get bounding boxes
    num_labels, labels_im = cv2.connectedComponents(bin_mask.astype(np.uint8))
    input_points = []
    input_boxes = []
    for label_idx in range(1, num_labels):
        ys, xs = np.where(labels_im == label_idx)
        x0, y0 = xs.min(), ys.min()
        x1, y1 = xs.max(), ys.max()

        # mass center as input point M[0][0], M[0][1]
        M = cv2.moments((labels_im == label_idx).astype(np.uint8))
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        input_point = [cX, cY]

        input_points.append(input_point)
        input_boxes.append([x0, y0, x1, y1])

    input_points = np.array(input_points)
    input_boxes = np.array(input_boxes)
    input_labels = np.array([1] * len(input_points))

    return input_points, input_labels, input_boxes
