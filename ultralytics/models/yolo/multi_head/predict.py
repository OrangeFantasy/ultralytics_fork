import torch

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, nms, ops


class MultiHeadPredictor(DetectionPredictor): 
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "multi-head"

    def postprocess(self, preds: torch.Tensor, img, orig_imgs, **kwargs):
        # Get head properties.
        m = self.model.model.model[-1]
        reg_max = m.reg_max
        heads = m.heads
        n_heads = m.n_heads
        nc_per_head = m.nc_per_head
        nc = sum(nc_per_head)
        nk = m.kpt_shape[0] * m.kpt_shape[1]
        n_pose_heads = m.n_pose_heads
        n_angles = m.n_angles
        n_angle_heads = m.n_angle_heads

        # Check confidence and IoU thresholds.
        conf_thresholds = self.args.conf
        if isinstance(conf_thresholds, (int, float)):
            conf_thresholds = [conf_thresholds] * n_heads
        assert len(conf_thresholds) == n_heads, f"Expected {n_heads} confidence thresholds, got {len(conf_thresholds)}"
        iou_thresholds = self.args.iou
        if isinstance(iou_thresholds, (int, float)):
            iou_thresholds = [iou_thresholds] * n_heads
        assert len(iou_thresholds) == n_heads, f"Expected {n_heads} IoU thresholds, got {len(iou_thresholds)}"

        # Split predictions by head.
        if isinstance(preds, (list, tuple)):
            preds = preds[0]  # Select only inference output.

        excepted_dims = (preds.shape[0], reg_max * 4 * n_heads + nc + nk * n_pose_heads + n_angles * n_angle_heads)
        assert preds.shape[1] == excepted_dims[1], f"Expected {excepted_dims[1]} predictions, got {preds.shape[1]}"

        boxes, scores, kpts, angles = preds.split((reg_max * 4 * n_heads, nc, nk * n_pose_heads, n_angles * n_angle_heads), dim=1)
        boxes_per_head  = list(boxes.chunk(n_heads, dim=1))
        scores_per_head = list(scores.split(nc_per_head, dim=1))
        kpts_per_head   = list(kpts.chunk(n_pose_heads, dim=1))
        angles_per_head = list(angles.chunk(n_angle_heads, dim=1))

        # Apply NMS to each head.
        batch_size, _, anchors = preds.shape
        dtype = preds.dtype

        preds_postprocessed = []
        for head_idx, head_type in reversed(enumerate(self.heads)):
            curr_head_preds = [boxes_per_head.pop(), scores_per_head.pop()]
            if head_type == "angle":
                curr_head_preds += [
                    torch.zeros((batch_size, nk, anchors), dtype=dtype, device=self.device),
                    angles_per_head.pop()
                ]
            elif head_type == "pose":
                curr_head_preds += [
                    kpts_per_head.pop(),
                    torch.zeros((batch_size, n_angles, anchors), dtype=dtype, device=self.device)
                ]
            elif head_type == "pose-angle":
                curr_head_preds += [
                    kpts_per_head.pop(),
                    angles_per_head.pop()
                ]
            else:
                curr_head_preds += [
                    torch.zeros((batch_size, nk, anchors), dtype=dtype, device=self.device),
                    torch.zeros((batch_size, n_angles, anchors), dtype=dtype, device=self.device)
                ]

            curr_head_preds = torch.cat(curr_head_preds, dim=1)
            curr_head_preds = nms.non_max_suppression(
                curr_head_preds,
                conf_thres=conf_thresholds[head_idx], 
                iou_thres=iou_thresholds[head_idx],
                nc=nc_per_head[head_idx],
                multi_label=True,
                max_det=self.args.max_det, 
                end2end=getattr(self.model, "end2end", False),
            )
            curr_head_preds[0][:, 5] += sum(nc_per_head[:head_idx])  # Adjust class indices.
            preds_postprocessed.append(curr_head_preds)

        assert len(boxes_per_head) == len(scores_per_head) == len(kpts_per_head) == len(angles_per_head) == 0

        # Concatenate predictions from different heads.
        preds = [torch.cat([x[0] for x in preds_postprocessed], 0)]

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list.
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        
        results = self.construct_results(preds, img, orig_imgs, **kwargs)
        return results

    def construct_result(self, pred, img, orig_img, img_path):
        result = super().construct_result(pred, img, orig_img, img_path)
        # Extract keypoints from prediction and reshape according to model's keypoint shape
        pred_kpts = pred[:, 6:-3].view(len(pred), *self.model.kpt_shape)
        pred_angs = pred[:, -3:].view(len(pred), 3)  # TODO: add angles in results
        # Scale keypoints coordinates to match the original image dimensions
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        result.update(keypoints=pred_kpts, angles=pred_angs)
        return result
