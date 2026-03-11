import os
import cv2
import tqdm
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from anolib.utils import refine_mask_basic
from anolib.visutils import get_inputs_from_mask, show_masks, get_overlap_img
from anolib.utils import postprocess_anomaly_mask
from anolib.ftbank import FeatureBank, stitch_sam2_patch_feats

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
LOG_CID = False
device = "cuda"


def debug(msg: str):
    if input(f"{msg} (q to quit)? ") == 'q':
        exit()


def init_env():
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        raise NotImplementedError("Only CUDA device is supported.")
    print(f"using device: {device}")


class InferModel:

    def __init__(self, model_cfg, model_ckpt, save_dir, infer_cdf='03'):
        self.model = build_sam2(model_cfg,
                                model_ckpt,
                                device=device,
                                save_dir=Path("exp/output_sam2"),
                                infer_cdf=infer_cdf)
        # Init Paths
        self.img_folder = Path(f"data/Eval/val/{infer_cdf}/test_images")
        self.gt_folder = Path(self.img_folder.parent / 'test_masks')

        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Init Predictor
        self.predictor = SAM2ImagePredictor(self.model)

    def _get_save_path(self, img_path: Path, is_anomaly_mask=False):
        if is_anomaly_mask:
            save_mask_path = Path(self.save_dir / 'anomaly_mask' /
                                  f'{img_path.stem}_anomaly_mask.png')
            save_mask_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            save_mask_path = Path(self.save_dir / f'{img_path.stem}.png')
        return save_mask_path

    def _get_grouped_imgpaths(self):
        img_list = sorted(list(self.img_folder.glob('*.jpg')))
        print(f'Found {len(img_list)} images for inference.')

        # Prepare CID(test, train) pairs
        cid_imgpaths = {}
        for img_path in img_list:
            parts = img_path.stem.split('@')
            cid = '@'.join(parts[0:-2])
            test_train = parts[-2]
            if cid not in cid_imgpaths:
                cid_imgpaths[cid] = {'test': [], 'train': []}
            cid_imgpaths[cid][test_train].append(img_path)

        # Debug
        cnt_unpaired = 0
        for cid, img_paths in cid_imgpaths.items():
            if len(img_paths['test']) == 0 or len(img_paths['train']) == 0:
                cnt_unpaired += 1
        cnt_pair = len(cid_imgpaths) - cnt_unpaired
        print(f'  Total CIDs: {len(cid_imgpaths)}, Paired CIDs: {cnt_pair}, ')
        return cid_imgpaths

    def _infer_img(self, img_path: Path):
        mask_path = self.gt_folder / f'{img_path.stem}.png'

        input_points, input_labels, input_boxes = get_inputs_from_mask(
            mask_path)

        image = Image.open(img_path)
        image = np.array(image.convert("RGB"))
        self.predictor.set_image(image)

        # [B, 3, H, W], low_f = [B, 256, H/16, W/16] = [1, 256, 64, 64]
        # high_f, low_f = self.predictor._features["high_res_feats"][0].detach().cpu().float().numpy(), \
        #                 self.predictor._features["image_embed"].detach().cpu().float().numpy()
        # visualiza_feature_map(high_f, img_path=img_path, bi=0)

        # print('# Multiple single box')
        boxes_mask = np.zeros((3, image.shape[0], image.shape[1]))
        boxes_scores = np.array([])
        for input_pt, input_lb, input_b in zip(input_points, input_labels,
                                               input_boxes):
            masks, scores, _ = self.predictor.predict(
                # point_coords=input_pt[None, ],
                # point_labels=input_lb[None, ],
                box=input_b[None, :],
                # mask_input=mask_input[None, :, :],
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]  # descending order
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]

            # add to overlap
            boxes_mask = np.maximum(boxes_mask, masks)
            boxes_scores = np.concatenate([boxes_scores, scores], axis=0)

        if len(boxes_scores) > 0:
            boxes_scores = boxes_scores.reshape(len(input_points), -1)
            boxes_scores = boxes_scores.mean(axis=0)

            best_idx = np.argmax(boxes_scores)
            best_mask = boxes_mask[best_idx]
            best_score = boxes_scores[best_idx]
            assert best_mask.shape == image.shape[:2]
            best_mask = refine_mask_basic(image, best_mask)
        else:
            best_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            best_score = 0.0

        save_mask_path = Path(self.save_dir / f'{img_path.stem}.png')
        cv2.imwrite(str(save_mask_path), best_mask.astype(np.uint8))
        show_masks(img_path, [best_mask],
                   scores=None,
                   point_coords=None,
                   input_labels=None,
                   box_coords=input_boxes,
                   borders=True,
                   mode='singlemask+multiboxes',
                   save_path=Path(save_mask_path).with_suffix('.jpg'))
        return best_mask, best_score

    def _infer_patch(self,
                     img_path: Path,
                     use_box: bool = True,
                     patch_size: int = 1024,
                     overlap: int = 100,
                     save_mask_path=None):

        # --- Load image ---
        image = Image.open(img_path)
        image = np.array(image.convert("RGB"))
        H, W = image.shape[:2]

        # --- Load corresponding mask ---
        mask_path = self.gt_folder / f'{img_path.stem}.png'

        if use_box:
            input_points, input_labels, input_boxes = get_inputs_from_mask(
                mask_path)

        # --- Prepare output placeholders ---
        full_mask = np.zeros((H, W), dtype=np.float32)

        # ---- helper: predict mask for given boxes ----
        def _predict_mask_for_boxes(rgb_img: np.ndarray,
                                    boxes: list,
                                    desc: str = ""):
            h, w = rgb_img.shape[:2]
            feats = None
            if len(boxes) == 0:
                return np.zeros((h, w), dtype=np.uint8), 0.0, feats

            self.predictor.set_image(rgb_img)
            try:
                box_inp = np.array(boxes) if use_box else None
                masks, scores, _ = self.predictor.predict(
                    box=box_inp,
                    multimask_output=True,
                )
                # masks=[N, H, W] or [H, W] -> masks=[H, W]
                masks = np.max(masks, axis=0)
                scores = np.mean(scores, axis=0)

                if scores.shape == ():
                    sel_mask = masks
                    sel_score = 0.0
                else:
                    best_idx = np.argmax(scores)
                    sel_mask = masks[best_idx]
                    sel_score = scores[best_idx]
                feats = self.predictor.get_feat_list()

            except Exception as e:
                print(f'Predict failed for {desc or img_path}, '
                      f'return empty mask. err={e}')
                sel_mask = np.zeros((h, w), dtype=np.uint8)
                sel_score = 0.0

            return sel_mask, sel_score, feats

        def _build_patch_regions(H: int, W: int, patch_size: int,
                                 overlap: int):
            # Case 1: if image is smaller than 2*patch_size, use single patch
            if H < 2 * patch_size and W < 2 * patch_size:
                return [(0, 0, W, H)]

            # Case2 : build patch regions
            step = patch_size - overlap
            y_starts = list(range(0, H, step))
            x_starts = list(range(0, W, step))

            if y_starts and y_starts[-1] + step > H:
                y_starts[-1] = max(H - step, 0)
            if x_starts and x_starts[-1] + step > W:
                x_starts[-1] = max(W - step, 0)

            patch_regions = []
            for y1 in y_starts:
                for x1 in x_starts:
                    y2 = min(y1 + patch_size, H)
                    x2 = min(x1 + patch_size, W)
                    patch_regions.append((x1, y1, x2, y2))

            return patch_regions

        def _assign_boxes_to_patches(input_boxes: list, patch_regions: list):
            """
            input_boxes: global coord [x0, y0, x1, y1]
            patch_regions: list of (px0, py0, px1, py1)
            """
            patch_boxes = [[] for _ in range(len(patch_regions))]

            for box in input_boxes:
                x0, y0, x1, y1 = box

                for idx, (px0, py0, px1, py1) in enumerate(patch_regions):
                    pw = px1 - px0
                    ph = py1 - py0

                    # patch-coords
                    b0 = x0 - px0
                    b1 = y0 - py0
                    b2 = x1 - px0
                    b3 = y1 - py0
                    # continue if box is outside patch
                    if b2 < 0 or b3 < 0 or b0 > pw or b1 > ph:
                        continue
                    # clip to patch boundary
                    b0 = max(0, b0)
                    b1 = max(0, b1)
                    b2 = min(pw, b2)
                    b3 = min(ph, b3)

                    patch_boxes[idx].append([b0, b1, b2, b3])

            return patch_boxes

        # --- Build patch regions and assign boxes ---
        patch_regions = _build_patch_regions(H, W, patch_size, overlap)
        patch_boxes = _assign_boxes_to_patches(
            input_boxes, patch_regions) if use_box else []

        patch_scores = []
        for idx, (x1, y1, x2, y2) in enumerate(patch_regions):
            patch = image[y1:y2, x1:x2, :]

            pbox_inp = patch_boxes[idx] if use_box else []
            patch_mask, patch_score, patch_feat = _predict_mask_for_boxes(
                patch, pbox_inp, desc=f'patch {idx} of {img_path}')

            patch_mask = refine_mask_basic(patch, patch_mask)
            full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2],
                                                 patch_mask)
            patch_scores.append(patch_score)

        # --- Normalize ---
        full_mask = (full_mask > 0).astype(np.uint8) * 255
        patch_score = np.array(patch_scores).mean() if len(
            patch_scores) > 0 else 0.0

        # --- Save visualization ---
        if save_mask_path is None:
            save_mask_path = self._get_save_path(img_path)

        cv2.imwrite(str(save_mask_path), full_mask)

        input_boxes = input_boxes if use_box else []
        show_masks(img_path, [full_mask],
                   scores=None,
                   point_coords=None,
                   input_labels=None,
                   box_coords=input_boxes,
                   borders=True,
                   mode='singlemask+multiboxes',
                   save_path=Path(save_mask_path).with_suffix('.jpg'))

        return full_mask

    def infer_imgs(
        self,
        use_box=True,
        ad_task=True,
    ):
        # 1. Prepare CID(test, train) pairs
        cid_imgpaths = self._get_grouped_imgpaths()

        # 2. Inference each CID
        cnt_debug = 0
        for i, (cid, img_paths) in enumerate(tqdm.tqdm(cid_imgpaths.items())):
            test_img_list = img_paths['test']
            train_img_list = img_paths['train']

            #--------------------------------------------------
            # DEBUG cid@test@000.jpg
            _img_path = Path(str(test_img_list[0])[:-7] + '000.jpg')
            _anomaly_save_path = self._get_save_path(_img_path,
                                                     is_anomaly_mask=True)
            _anomaly_rawfeat_save_path = _anomaly_save_path.with_suffix(
                '.rawfeat.png')
            # --------------------------------------------------
            if len(test_img_list) == 0 or len(train_img_list) == 0:
                continue  # skip unpaired CIDs

            example_img = Image.open(train_img_list[0])
            imgW, imgH = example_img.size

            # Debug
            # cnt_debug += 1
            # if cnt_debug > 20:
            #     exit()

            # ------ 1. For CID Segmentation ------
            if LOG_CID:
                print(f'[Segmentation] CID {i+1}/{len(cid_imgpaths)}: {cid}')
            cid_all_component_mask = np.zeros((imgH, imgW), dtype=np.uint8)
            # 1-1. Inference train image
            cur_patch_size = 1024
            for img_path in train_img_list:
                best_mask = self._infer_patch(img_path=img_path,
                                              use_box=use_box,
                                              patch_size=cur_patch_size)
                cid_all_component_mask = np.maximum(cid_all_component_mask,
                                                    best_mask)

                if best_mask.max() == 0:
                    best_mask = self._infer_patch(img_path=img_path,
                                                  use_box=use_box,
                                                  patch_size=cur_patch_size)
                    cid_all_component_mask = np.maximum(
                        cid_all_component_mask, best_mask)
            # 1-2. Inference test image
            for img_path in test_img_list:
                best_mask = self._infer_patch(img_path=img_path,
                                              use_box=use_box,
                                              patch_size=cur_patch_size)
                cid_all_component_mask = np.maximum(cid_all_component_mask,
                                                    best_mask)
                if best_mask.max() == 0:
                    best_mask = self._infer_patch(img_path=img_path,
                                                  use_box=use_box,
                                                  patch_size=cur_patch_size)
                    cid_all_component_mask = np.maximum(
                        cid_all_component_mask, best_mask)

            # ------ 2. For AD Task ------
            if ad_task:
                # 2-1. Build feature bank from train images
                if LOG_CID:
                    print(
                        f'[Anomaly Detection] CID {i+1}/{len(cid_imgpaths)}: {cid}'
                    )
                train_embeds = []
                for img_path in train_img_list:
                    image = Image.open(img_path)
                    image = np.array(image.convert("RGB"))
                    full_feats = stitch_sam2_patch_feats(self.predictor,
                                                         full_image=image,
                                                         patch_size=1024)
                    train_embeds.append(full_feats)
                bank = FeatureBank(train_embeds)

                # 2-2. Inference each test image
                for img_path in test_img_list:
                    image = Image.open(img_path)
                    image = np.array(image.convert("RGB"))
                    full_feats = stitch_sam2_patch_feats(self.predictor,
                                                         full_image=image,
                                                         patch_size=1024)

                    anomaly_save_path = self._get_save_path(
                        img_path, is_anomaly_mask=True)
                    anomaly_feat_save_path = anomaly_save_path.with_suffix(
                        '.feat.png')

                    anomaly_map = bank.cal_anomaly_map(full_feats)
                    anomaly_map = cv2.resize(anomaly_map,
                                             (image.shape[1], image.shape[0]))

                    # visualiza_feature_map(anomaly_map,
                    #                       img_path=img_path,
                    #                       bi=0,
                    #                       save_path=anomaly_feat_save_path,
                    #                       instance_mask=cid_all_component_mask,
                    #                       norm=True)
                    # Raw anomaly map
                    anomaly_rawmap_gray = (anomaly_map * 255).clip(
                        0, 255).astype(np.uint8)
                    cv2.imwrite(
                        str(anomaly_feat_save_path).replace(
                            '.feat', '.rawfeat'), anomaly_rawmap_gray)

                    # Raw anomaly map + Instance Masking
                    anomaly_map_gray = anomaly_map * (cid_all_component_mask
                                                      > 0)
                    anomaly_map_gray = (anomaly_map_gray * 255).clip(
                        0, 255).astype(np.uint8)
                    cv2.imwrite(str(anomaly_feat_save_path), anomaly_map_gray)

                    # Thresholding + Instance Masking
                    anomaly_map = postprocess_anomaly_mask(anomaly_map,
                                                           thresh=99)

                    # Binarize anomaly map
                    raw_binary_anomaly_save_path = str(
                        anomaly_save_path).replace('_anomaly_mask.png',
                                                   '_anomaly_rawmask.png')
                    cv2.imwrite(raw_binary_anomaly_save_path, anomaly_map)

                    # Instance Masked Binary Anomaly Map
                    anomaly_map = anomaly_map * (cid_all_component_mask > 0)
                    cv2.imwrite(str(anomaly_save_path), anomaly_map)

                    # Overlap Visualization
                    overlap_anomaly = get_overlap_img(img_path, anomaly_map)
                    cv2.imwrite(str(anomaly_save_path.with_suffix('.jpg')),
                                overlap_anomaly)

        print(f'Saved results to {self.save_dir}')


class InferCDFModel(InferModel):

    def __init__(
        self,
        model_cfg,
        model_ckpt,
        save_dir,
        # additional args for InferCDF
        infer_img_folder,
    ):
        super().__init__(model_cfg, model_ckpt, save_dir)
        self.img_folder = infer_img_folder
        self.gt_folder = Path(self.img_folder.parent /
                              self.img_folder.name.replace('images', 'masks'))


class BaseConfig():

    def __init__(self):
        pass

    @staticmethod
    def _stripped_model_type(model_type: str) -> str:
        # Customize
        stripped = model_type
        stripped = stripped.split('(')[0]
        return stripped

    def get_sam_cfg(self, model_type: str) -> str:
        BASE_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
        if model_type == "0SHOT":
            return BASE_CFG
        
        # Customize
        # Let model_type be "LORA_r16(0207051004)", we strip it to "LORA_r16" 
        # and find corresponding config "sam2.1_hiera_l_LORA_r16.yaml"
        stripped = self._stripped_model_type(model_type)
        config = f"configs/sam2.1/sam2.1_hiera_l_{stripped}.yaml"
        return config

    def get_sam_ckpt(self, model_type: str, epoch: str) -> str:
        # Customize
        BASE_CKPT = "checkpoints/sam2.1_hiera_large.pt"
        if model_type == "0SHOT":
            return BASE_CKPT
        sam2_checkpoint = f"sam2_logs/{model_type}/checkpoints/checkpoint_{epoch}.pt"
        return sam2_checkpoint


def demo():
    img_path = 'data/infer.jpg'
    model_cfg = 'configs/sam2.1/sam2.1_hiera_l_DORA_r16.yaml'
    model_ckpt = 'sam2_logs/DORA_r16(0207051004_stage2)/checkpoints/checkpoint_100.pt'
    model = build_sam2(model_cfg, model_ckpt, device=device)

    image = Image.open(img_path)
    image = np.array(image.convert("RGB"))
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(image)

    boxes_mask = np.zeros((3, image.shape[0], image.shape[1]))
    boxes_scores = np.array([])
    boxes_mask, boxes_scores, _ = predictor.predict(
        box=[0, 0, image.shape[1] - 1, image.shape[0] - 1],
        # point_coords=input_pt[None, ],
        # point_labels=input_lb[None, ],
        # mask_input=mask_input[None, :, :],
        multimask_output=True,
    )
    print(f'shape boxes_mask: {boxes_mask.shape}')
    print(f'boxes_scores: {boxes_scores}')
    if len(boxes_scores) > 0:
        # boxes_scores = boxes_scores.reshape(len(input_points), -1)
        # boxes_scores = boxes_scores.mean(axis=0)
        best_idx = np.argmax(boxes_scores)
        best_mask = boxes_mask[best_idx]
        # best_score = boxes_scores[best_idx]
        assert best_mask.shape == image.shape[:2]
        best_mask = refine_mask_basic(image, best_mask)
    else:
        best_mask = boxes_mask[0]
        best_score = boxes_scores[0]
        # best_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # best_score = 0.0

    save_mask_path = Path(img_path).parent / f"{Path(img_path).stem}_demo.png"
    cv2.imwrite(str(save_mask_path), best_mask.astype(np.uint8))
    show_masks(img_path, [best_mask],
               scores=None,
               point_coords=None,
               input_labels=None,
               box_coords=[],
               borders=True,
               mode='singlemask+multiboxes',
               save_path=Path(save_mask_path).with_suffix('.jpg'))


if __name__ == "__main__":
    demo()