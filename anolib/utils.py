import cv2
import torch
import numpy as np
from typing import Iterable, Literal, Optional, Union, List, Tuple


# =============================================================================================================
def visualiza_feature_map(
    img_feature: torch.Tensor,
    img_path,
    bi,
    save_path=None,
    instance_mask=None,
    norm=True,
):
    # img_feature: [b, c, h, w]
    if isinstance(img_feature, torch.Tensor):
        img_feature = img_feature.detach().cpu().numpy()
    print(f"  feature map shape: {img_feature.shape}")
    if img_feature.ndim == 4:
        img_feature = img_feature[0].transpose(1, 2, 0)  # [h,w,c]
    if img_feature.ndim == 3:
        img_feature = img_feature.mean(axis=2)  # [h,w]

    # Assert values in [0, 1]
    # img_feature_heatmap = (img_feature - img_feature.min()) / (
    #     img_feature.max() - img_feature.min() + 1e-8)
    # img_feature_heatmap = (img_feature_heatmap * 255).astype(np.uint8)
    if norm:
        assert (
            img_feature.min() >= 0 and img_feature.max() <= 1
        ), "When norm=False, the feature map values must be in [0, 1]."
        img_feature_heatmap = img_feature
        img_feature_heatmap = (img_feature_heatmap * 255).astype(np.uint8)
    else:
        assert (
            img_feature.min() >= 0 and img_feature.max() <= 255
        ), "When norm=False, the feature map values must be in [0, 255]."
        img_feature_heatmap = img_feature.astype(np.uint8)
    if instance_mask is not None:
        img_feature_heatmap = img_feature_heatmap * (instance_mask > 0)
    print(f"max val: {img_feature_heatmap.max()}, min val: {img_feature_heatmap.min()}")
    img_feature_heatmap = cv2.applyColorMap(img_feature_heatmap, cv2.COLORMAP_JET)
    img_ori = cv2.imread(img_path)
    img_feature_heatmap = cv2.resize(
        img_feature_heatmap, (img_ori.shape[1], img_ori.shape[0])
    )
    if save_path is None:
        save_path = f"test_img_feature_{bi}.jpg"
    cv2.imwrite(save_path, img_feature_heatmap)


# =============================================================================================================
def postprocess_anomaly_mask(
    mask: np.ndarray,
    thresh: Union[Literal["otsu"], float] = "otsu",
    keep="all",  # "all" | "largest"
):

    # Input assertion
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
        mask = np.clip(mask, 0, 255)

    # Light denoise; median is edge-preserving and usually enough
    m = cv2.medianBlur(mask, 5)

    # Thresholding
    bin_m = None
    if isinstance(thresh, (int, float)):
        _, bin_m = cv2.threshold(m, float(thresh), 255, cv2.THRESH_BINARY)

    elif thresh == "std":
        bin_m = (m > (m.mean() + 4 * m.std())).astype(np.uint8) * 255
    elif thresh == "mad":
        # median absolute deviation
        bin_m = (m > (np.median(m) + 4 * np.median(np.abs(m - np.median(m))))).astype(
            np.uint8
        ) * 255
    elif thresh == "otsu":
        _, bin_m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thresh == "mix":
        # try 99.9, std, mad, otsu in order, and use the one with minimum connected components but > 1
        bin_m_std = (m > (m.mean() + 4 * m.std())).astype(np.uint8) * 255
        bin_m_mad = (
            m > (np.median(m) + 4 * np.median(np.abs(m - np.median(m))))
        ).astype(np.uint8) * 255
        _, bin_m_otsu = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, bin_m_999 = cv2.threshold(m, 0.999 * 255, 255, cv2.THRESH_BINARY)
        bin_ms = [bin_m_999, bin_m_std, bin_m_mad, bin_m_otsu]
        bin_m = None
        min_num_labels = float("inf")
        for bm in bin_ms:
            num_labels, _, _, _ = cv2.connectedComponentsWithStats(
                (bm > 0).astype(np.uint8), connectivity=8
            )
            if 1 < num_labels < min_num_labels:
                min_num_labels = num_labels
                bin_m = bm
    else:
        raise ValueError("thresh must be 'otsu', 'std', 'mad', or a numeric value.")

    # If bin_m is all zeros, fallback to Otsu
    if bin_m is not None and bin_m.max() == 0:
        _, bin_m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological : close for small holes ; open for small objects
    for k_size in [5, 7]:
        K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        bin_m = cv2.morphologyEx(bin_m, cv2.MORPH_CLOSE, K, iterations=1)
        bin_m = cv2.morphologyEx(bin_m, cv2.MORPH_OPEN, K, iterations=1)

    # Connected components analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (bin_m > 0).astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return (bin_m > 0).astype(np.uint8) * 255

    # Largest or min area filtering
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip bg label 0
    labels_idx = np.arange(1, num_labels)
    if keep == "largest":
        keep_labels = labels_idx[np.argmax(areas)][None]
    else:
        min_area = 30
        keep_labels = labels_idx[areas >= int(min_area)]

    # Return binary mask
    keep_map = np.isin(labels, keep_labels).astype(np.uint8) * 255
    return keep_map


# =============================================================================================================
def resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    # Resize
    if not mask.shape[:1] == (h, w):
        interpolation_method = (
            cv2.INTER_NEAREST if len(np.unique(mask)) == 2 else cv2.INTER_LINEAR
        )
        mask_img = cv2.resize(mask, (w, h), interpolation=interpolation_method)
    return mask_img


def path_to_ndarray(path, gray=False):
    if isinstance(path, str):
        arr = cv2.imread(path)
        if arr.ndim == 2 or gray:
            arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return arr
    elif isinstance(path, np.ndarray):
        return path
    else:
        raise TypeError("path must be str or np.ndarray")


def prepare_img_mask(img_path, mask_path, gray=False, return_mask_size=False):

    mask = path_to_ndarray(mask_path, gray=gray)
    if img_path is not None:
        img = path_to_ndarray(img_path)
    else:
        img = img_path
        print("Warning: img_path is None, using img_path as ndarray directly.")
        return img, mask

    if not return_mask_size and mask.shape[:2] != img.shape[:2]:
        mask = resize_mask(mask, img.shape[0], img.shape[1])
    return img, mask


def refine_mask_basic(img_path, mask_path, return_mask_size=False):
    img, mask = prepare_img_mask(img_path, mask_path, return_mask_size=return_mask_size)
    # Operations
    # Remove small holes
    original_mask_shape = mask.shape

    refined = mask.copy()
    for k in [3, 5]:
        K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # open : remove small object; close: remove small holes
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, K, iterations=1)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, K, iterations=1)

    # Smooth edges
    refined = cv2.medianBlur(refined, 3)
    ref_f = refined.astype(np.float32) / 255.0
    ref_f = cv2.GaussianBlur(ref_f, (3, 3), 0)
    # ref_f = cv2.bilateralFilter(ref_f, d=7, sigmaColor=ref_f.std()*0.5, sigmaSpace=5)
    refined = (ref_f > 0).astype(np.uint8) * 255

    # Binarize to [0, 255]
    # cv2.imwrite('output/refined_mask.png', refined_m * 255)
    assert (
        refined.shape == original_mask_shape
    ), f"Refined mask shape {refined.shape} does not match original mask shape"
    return refined
