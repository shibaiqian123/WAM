"""Native SONIC latent / Dex1 interface, isolated from physical-action models."""
import copy
import torch
from .wbwam_optional_idm import WBWAMOptionalIDM
from wbwam.utils.logging_config import get_logger

logger = get_logger(__name__)
CONTRACT = {
    "representation": "sonic_latent64_dex1_continuous_v1",
    "state_dim": 31, "action_dim": 66,
    "state_order": ["sonic_body_joints_29", "left_dex1_position", "right_dex1_position"],
    "action_order": ["sonic_motion_token_64", "left_dex1_target", "right_dex1_target"],
}
INTERFACE_KEYS = {
    "mixtures.action.action_encoder.weight", "mixtures.action.action_encoder.bias",
    "mixtures.action.head.weight", "mixtures.action.head.bias",
}

class WBWAMSonicLatent(WBWAMOptionalIDM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.proprio_dim != 31 or self.action_expert.action_dim != 66:
            raise ValueError("SONIC requires native proprio_dim=31 and action_dim=66")
        self._sonic_ready = False
        self._sonic_metrics = {}

    @property
    def checkpoint_contract(self):
        return copy.deepcopy(CONTRACT)

    def complete_training_state_restore(self, contract):
        if contract != CONTRACT:
            raise ValueError("SONIC training-state contract mismatch")
        self._sonic_ready = True

    def _require_ready(self):
        if not self._sonic_ready:
            raise RuntimeError("Load a complete SONIC checkpoint or initialize from physical midtrain first")

    @staticmethod
    def _check_state(expected, actual, label):
        if set(expected) != set(actual):
            raise ValueError(f"{label} keys mismatch: missing={sorted(set(expected)-set(actual))}, unexpected={sorted(set(actual)-set(expected))}")
        for key in expected:
            if not isinstance(actual[key], torch.Tensor) or expected[key].shape != actual[key].shape:
                raise ValueError(f"{label} shape mismatch: {key}")

    def initialize_from_checkpoint(self, path):
        self._sonic_ready = False
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if "sonic_contract" in payload:
            raise ValueError("Use resume/load_checkpoint for a SONIC checkpoint")
        source = payload.get("mot", {})
        target = self.mot.state_dict()
        if not INTERFACE_KEYS <= set(target):
            raise ValueError("Unexpected SONIC interface layout")
        self._check_state({k:v for k,v in target.items() if k not in INTERFACE_KEYS},
                          {k:v for k,v in source.items() if k not in INTERFACE_KEYS}, "MoT backbone")
        # Validate the complete physical source interface, including unchanged-size biases.
        expected_shapes = {
            "mixtures.action.action_encoder.weight": (self.action_expert.hidden_dim, 96),
            "mixtures.action.action_encoder.bias": (self.action_expert.hidden_dim,),
            "mixtures.action.head.weight": (96, self.action_expert.hidden_dim),
            "mixtures.action.head.bias": (96,),
        }
        for key, shape in expected_shapes.items():
            if key not in source or tuple(source[key].shape) != shape:
                raise ValueError(f"Physical checkpoint interface mismatch: {key}")
        state = payload.get("proprio_encoder", {})
        if set(state) != {"weight", "bias"} or tuple(state["weight"].shape) != (self.text_dim, 96) or tuple(state["bias"].shape) != (self.text_dim,):
            raise ValueError("Physical checkpoint must contain complete 96D proprio projection")
        # Initialize on CPU for deterministic results independent of destination device.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            for layer in (self.action_expert.action_encoder, self.action_expert.head, self.proprio_encoder):
                fresh = torch.nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None)
                layer.load_state_dict(fresh.state_dict(), strict=True)
        merged = self.mot.state_dict()
        merged.update({k:v for k,v in source.items() if k not in INTERFACE_KEYS})
        self.mot.load_state_dict(merged, strict=True)
        self._sonic_ready = True
        report = {"loaded_backbone_tensors": len(source)-len(INTERFACE_KEYS),
                  "reinitialized": sorted(INTERFACE_KEYS) + ["proprio_encoder.weight", "proprio_encoder.bias"],
                  "source_step": payload.get("step"), "training_step": 0}
        logger.info("SONIC migration: %s", report)
        return report

    def save_checkpoint(self, path, optimizer=None, step=None):
        self._require_ready()
        payload = {"mot": self.mot.state_dict(), "proprio_encoder": self.proprio_encoder.state_dict(),
                   "sonic_contract": copy.deepcopy(CONTRACT), "step": step, "torch_dtype": str(self.torch_dtype)}
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None, strict=None):
        self._sonic_ready = False
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("sonic_contract") != CONTRACT:
            raise ValueError("SONIC checkpoint contract mismatch; physical weights require init_from_checkpoint")
        self._check_state(self.mot.state_dict(), payload.get("mot", {}), "MoT")
        self._check_state(self.proprio_encoder.state_dict(), payload.get("proprio_encoder", {}), "proprio")
        self.mot.load_state_dict(payload["mot"], strict=True)
        self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        self._sonic_ready = True
        return payload

    def _compute_action_loss_per_sample(self, pred_action, target_action, action_is_pad, action_dim_is_pad):
        if pred_action.shape != target_action.shape or pred_action.ndim != 3 or pred_action.shape[-1] != 66:
            raise ValueError("SONIC action prediction and target must have matching [B,H,66] shapes")
        valid = torch.ones_like(pred_action, dtype=torch.bool)
        if action_is_pad is not None:
            if action_is_pad.shape != pred_action.shape[:2]:
                raise ValueError("Temporal padding shape mismatch")
            valid &= ~action_is_pad.bool().unsqueeze(-1)
        if action_dim_is_pad is not None:
            if action_dim_is_pad.shape != pred_action.shape:
                raise ValueError("Dimension padding shape mismatch")
            valid &= ~action_dim_is_pad.bool()
        diff = (pred_action.float()-target_action.float()).masked_fill(~valid, 0)
        errors = diff.square()
        def reduce(e, mask):
            return e.sum(dim=(1,2)) / mask.sum(dim=(1,2)).clamp_min(1)
        self._sonic_metrics = {
            "sonic_latent_flow_mse": float(reduce(errors[:,:,:64],valid[:,:,:64]).mean().detach()),
            "dex1_flow_mse": float(reduce(errors[:,:,64:],valid[:,:,64:]).mean().detach()),
        }
        return reduce(errors, valid)

    def training_loss(self, *args, **kwargs):
        self._require_ready()
        if getattr(self, "fk_link_loss", None) is not None:
            raise ValueError("FK loss is not supported for SONIC latent actions")
        loss, metrics = super().training_loss(*args, **kwargs)
        metrics.update(self._sonic_metrics)
        return loss, metrics

    def infer_action(self, *args, **kwargs):
        self._require_ready()
        return super().infer_action(*args, **kwargs)

    def infer_joint(self, *args, **kwargs):
        self._require_ready()
        return super().infer_joint(*args, **kwargs)

    def infer_action_no_video(self, *args, **kwargs):
        self._require_ready()
        return super().infer_action_no_video(*args, **kwargs)

def create_sonic_latent(**kwargs):
    from wbwam.runtime import _create_wbwam_idm_from_cls
    return _create_wbwam_idm_from_cls(WBWAMSonicLatent, **kwargs)
