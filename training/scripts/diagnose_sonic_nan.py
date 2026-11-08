"""Bounded real-data NaN localization; no long training or checkpoint writes."""
from pathlib import Path
import argparse
import torch
from omegaconf import OmegaConf
from wbwam.trainer import Trainer
from wbwam.runtime import run_training
from wbwam.utils.config_resolvers import register_default_resolvers

def check(label, value):
    if torch.is_tensor(value) and value.is_floating_point():
        if not torch.isfinite(value).all():
            raise RuntimeError("NONFINITE: " + label)

original_init = Trainer.__init__
def diagnostic_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    model = self.accelerator.unwrap_model(self.model)
    rank = self.accelerator.process_index
    original_loss = model.training_loss
    calls = 0
    def loss(sample, *a, **kw):
        nonlocal calls
        calls += 1
        for key, value in sample.items():
            check("input/" + key, value)
        print(f"DIAG rank={rank} micro={calls} inputs finite", flush=True)
        result = original_loss(sample, *a, **kw)
        check("forward/loss", result[0])
        print(f"DIAG rank={rank} micro={calls} loss={float(result[0].detach())} metrics={result[1]}", flush=True)
        return result
    model.training_loss = loss
    for name, module in [("action_input", model.action_expert.action_encoder),
                         ("action_output", model.action_expert.head),
                         ("state_input", model.proprio_encoder)]:
        for pn, param in module.named_parameters():
            def grad_hook(grad, label=name + "/" + pn):
                check("gradient/" + label, grad)
                return grad
            param.register_hook(grad_hook)
    engine = self.model
    if hasattr(engine, "optimizer") and hasattr(engine.optimizer, "optimizer"):
        optimizer = engine.optimizer.optimizer
        original_step = optimizer.step
        def step(*a, **kw):
            for gi, group in enumerate(optimizer.param_groups):
                for pi, param in enumerate(group["params"]):
                    check(f"before_optimizer/param/{gi}/{pi}", param)
                    if param.grad is not None:
                        check(f"before_optimizer/grad/{gi}/{pi}", param.grad)
            for group in optimizer.param_groups:
                print("DIAG optimizer options", {k:v for k,v in group.items() if k != "params"}, flush=True)
                for param in group["params"]:
                    print("DIAG param", param.dtype, param.numel(), "grad range",
                          float(param.grad.min()), float(param.grad.max()), flush=True)
                    for key, value in optimizer.state[param].items():
                        if torch.is_tensor(value):
                            print("DIAG state before", key, value.dtype, value.numel(),
                                  float(value.min()), float(value.max()), flush=True)
            result = original_step(*a, **kw)
            for group in optimizer.param_groups:
                for param in group["params"]:
                    for key, value in optimizer.state[param].items():
                        if torch.is_tensor(value):
                            print("DIAG state after", key, value.dtype, value.numel(),
                                  float(value.min()), float(value.max()), flush=True)
            for gi, group in enumerate(optimizer.param_groups):
                for pi, param in enumerate(group["params"]):
                    check(f"after_optimizer/param/{gi}/{pi}", param)
            print(f"DIAG rank={rank} optimizer update finite", flush=True)
            return result
        optimizer.step = step
Trainer.__init__ = diagnostic_init

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    register_default_resolvers()
    cfg = OmegaConf.load(args.config)
    cfg.output_dir = args.output
    cfg.max_steps = 2
    cfg.log_every = 1
    cfg.save_every = 0
    cfg.save_final = False
    cfg.eval_every = 0
    cfg.eval_at_start = False
    cfg.num_workers = 0
    cfg.tensorboard.enabled = False
    cfg.wandb.enabled = False
    run_training(cfg)
