import os
import unittest
from unittest import mock

import torch

from wbwam.models.wan22.wbwam import WBWAM


class _VAEModel:
    def __init__(self):
        self.calls = 0

    def encode(self, value, scale):
        self.calls += 1
        return value * scale


class _VAE:
    def __init__(self):
        self.model = _VAEModel()
        self.scale = 2.0


class VAEEncodeModeTest(unittest.TestCase):
    def setUp(self):
        self.model = WBWAM.__new__(WBWAM)
        self.model.device = torch.device("cpu")
        self.model.vae = _VAE()

    def test_eager_environment_bypasses_torch_compile(self):
        value = torch.ones(1, 3, 1, 2, 2)
        with mock.patch.dict(os.environ, {"WBWAM_DISABLE_VAE_COMPILE": "1"}), mock.patch(
            "torch.compile", side_effect=AssertionError("torch.compile must not run")
        ):
            result = self.model._encode_video_latents(value)
        self.assertEqual(self.model.vae.model.calls, 1)
        self.assertFalse(hasattr(self.model, "_vae_encode_compiled"))
        torch.testing.assert_close(result, value * 2)

    def test_default_path_still_uses_compile(self):
        value = torch.ones(1, 3, 1, 2, 2)
        compiled = mock.Mock(side_effect=self.model.vae.model.encode)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WBWAM_DISABLE_VAE_COMPILE", None)
            with mock.patch("torch.compile", return_value=compiled) as compile_mock:
                result = self.model._encode_video_latents(value)
        compile_mock.assert_called_once()
