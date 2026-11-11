"""Frozen SONIC decoder and FK auxiliary supervision for native actions."""

from __future__ import annotations

import torch


class SonicEndEffectorLoss:
    """Decode clean 64D SONIC tokens and compare frozen FK end effectors."""

    def __init__(
        self,
        decoder,
        normalizer,
        *,
        weight: float = 1.0,
        position_weight: float = 1.0,
        rotation_weight: float = 0.1,
    ):
        import math

        self.decoder = decoder.eval().requires_grad_(False)
        self.action_normalizer = normalizer.action_normalizer
        self.weight = float(weight)
        self.position_weight = float(position_weight)
        self.rotation_weight = float(rotation_weight)
        if (
            not math.isfinite(self.weight)
            or self.weight < 0
            or not math.isfinite(self.position_weight)
            or not math.isfinite(self.rotation_weight)
            or self.position_weight < 0
            or self.rotation_weight < 0
        ):
            raise ValueError("SONIC EEF loss weights must be finite and non-negative")

    def to(self, device):
        self.decoder.to(device=device)
        return self

    def __call__(self, prediction, target, action_dim_is_pad):
        if prediction.shape != target.shape or prediction.shape != action_dim_is_pad.shape:
            raise ValueError("SONIC EEF loss expects matching [B,H,66] tensors")
        if prediction.ndim != 3 or prediction.shape[-1] != 66:
            raise ValueError("SONIC EEF loss expects [B,H,66] actions")
        valid = ~action_dim_is_pad[..., :64].bool().any(-1)
        if not valid.any():
            zero = prediction.float().sum() * 0.0
            return {"position_m2": zero, "rotation_rad2": zero, "weighted": zero}
        if not torch.isfinite(prediction[valid]).all() or not torch.isfinite(target[valid]).all():
            raise ValueError("Non-finite valid action in SONIC EEF loss")

        if self.action_normalizer.mode != "min/max":
            raise ValueError("SONIC EEF loss requires the configured min/max action normalizer")
        scale = self.action_normalizer.scale
        offset = self.action_normalizer.offset
        if scale.ndim == 2:
            if scale.shape[0] < prediction.shape[1]:
                raise ValueError("Action normalization stats are shorter than the action horizon")
            scale = scale[: prediction.shape[1]]
            offset = offset[: prediction.shape[1]]
            scale = scale.unsqueeze(0)
            offset = offset.unsqueeze(0)
        else:
            scale = scale.view(1, 1, -1)
            offset = offset.view(1, 1, -1)
        scale = scale[..., :64].to(device=prediction.device, dtype=torch.float32)
        offset = offset[..., :64].to(device=prediction.device, dtype=torch.float32)
        pred_raw = ((prediction.float()[..., :64].clamp(-1, 1) - offset) / scale)[valid].unsqueeze(0)
        target_raw = ((target.float()[..., :64].clamp(-1, 1) - offset) / scale)[valid].unsqueeze(0)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            pred_pos, pred_rot = self.decoder.decode_differentiable(pred_raw.float())
            with torch.no_grad():
                target_pos, target_rot = self.decoder.decode_differentiable(target_raw.float())
            position = (pred_pos - target_pos).square().sum(-1).mean()
            relative = pred_rot.transpose(-1, -2) @ target_rot
            skew = torch.stack(
                (
                    relative[..., 2, 1] - relative[..., 1, 2],
                    relative[..., 0, 2] - relative[..., 2, 0],
                    relative[..., 1, 0] - relative[..., 0, 1],
                ),
                dim=-1,
            )
            sine = torch.linalg.vector_norm(skew, dim=-1) * 0.5
            cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
            rotation = torch.atan2(sine, cosine).square().mean()
            weighted = self.weight * (
                self.position_weight * position + self.rotation_weight * rotation
            )
        return {
            "position_m2": position,
            "rotation_rad2": rotation,
            "weighted": weighted,
        }
