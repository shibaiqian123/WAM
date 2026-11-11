import unittest

import torch
from torch import nn

from wbwam.models.wan22.sonic_eef_loss import SonicEndEffectorLoss


class _ActionNormalizer:
    mode = "min/max"

    def __init__(self):
        self.scale = torch.ones(66)
        self.offset = torch.zeros(66)


class _Normalizer:
    action_normalizer = _ActionNormalizer()


class _FakeSonicDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def decode_differentiable(self, tokens):
        point = tokens[..., :3] * self.scale
        positions = torch.stack((point, point + tokens[..., 3:4]), dim=-2)
        angle = tokens[..., 4]
        zero, one = torch.zeros_like(angle), torch.ones_like(angle)
        cosine, sine = angle.cos(), angle.sin()
        row0 = torch.stack((cosine, -sine, zero), dim=-1)
        row1 = torch.stack((sine, cosine, zero), dim=-1)
        row2 = torch.stack((zero, zero, one), dim=-1)
        rotation = torch.stack((row0, row1, row2), dim=-2)
        rotations = rotation.unsqueeze(-3).expand(*rotation.shape[:-2], 2, 3, 3)
        return positions, rotations


class SonicEndEffectorLossTest(unittest.TestCase):
    def test_loss_matches_formula_and_routes_gradient_to_latent(self):
        decoder = _FakeSonicDecoder()
        criterion = SonicEndEffectorLoss(
            decoder,
            _Normalizer(),
            weight=0.05,
            position_weight=1.0,
            rotation_weight=0.1,
        )
        prediction = torch.zeros(1, 3, 66, requires_grad=True)
        target = torch.zeros_like(prediction)
        target[0, 0, 0] = 0.25
        target[0, 1, 4] = 0.2
        action_dim_is_pad = torch.zeros_like(prediction, dtype=torch.bool)
        action_dim_is_pad[:, 2] = True

        metrics = criterion(prediction, target, action_dim_is_pad)
        expected = 0.05 * (
            metrics["position_m2"] + 0.1 * metrics["rotation_rad2"]
        )
        self.assertTrue(torch.isfinite(metrics["weighted"]))
        self.assertAlmostEqual(metrics["weighted"].item(), expected.item(), places=6)
        metrics["weighted"].backward()

        self.assertGreater(prediction.grad[:, :2, :64].norm().item(), 0.0)
        self.assertEqual(prediction.grad[:, 2].abs().sum().item(), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in decoder.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in decoder.parameters()))

    def test_all_padded_batch_is_finite_zero(self):
        criterion = SonicEndEffectorLoss(_FakeSonicDecoder(), _Normalizer(), weight=0.05)
        prediction = torch.randn(1, 2, 66, requires_grad=True)
        target = torch.zeros_like(prediction)
        action_dim_is_pad = torch.ones_like(prediction, dtype=torch.bool)

        result = criterion(prediction, target, action_dim_is_pad)
        self.assertEqual(result["weighted"].item(), 0.0)
        result["weighted"].backward()
        self.assertEqual(prediction.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
