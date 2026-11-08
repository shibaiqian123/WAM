import unittest
import torch
from wbwam.trainer import _split_optimizer_param_groups

class OptimizerGroupsTest(unittest.TestCase):
    def test_order_identity_and_hyperparameters(self):
        params = [torch.nn.Parameter(torch.zeros(n)) for n in (4, 3, 5, 2)]
        groups = [{"params": params[:3], "lr": .01, "weight_decay": .1},
                  {"params": params[3:], "lr": .01, "weight_decay": 0.}]
        result = _split_optimizer_param_groups(groups, 7)
        self.assertEqual([sum(p.numel() for p in g["params"]) for g in result], [7,5,2])
        self.assertEqual([id(p) for g in result for p in g["params"]], [id(p) for p in params])
        self.assertEqual([g["weight_decay"] for g in result], [.1,.1,0.])
        self.assertTrue(all(g["lr"] == .01 for g in result))
        self.assertEqual(len(groups[0]["params"]), 3)
    def test_disabled_preserves_official_groups(self):
        groups = [{"params": []}]
        self.assertIs(_split_optimizer_param_groups(groups, None), groups)
    def test_invalid_cap_and_oversized_parameter(self):
        with self.assertRaises(ValueError):
            _split_optimizer_param_groups([], 0)
        with self.assertRaises(ValueError):
            _split_optimizer_param_groups([{"params":[torch.zeros(8)]}], 7)
