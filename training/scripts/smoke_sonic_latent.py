"""Bounded synthetic smoke: no robot, dataset conversion, or optimizer steps."""
import argparse
import json
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--report",required=True)
    args=parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Select one available A100 using CUDA_VISIBLE_DEVICES")
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/"configs"),version_base=None):
        cfg=compose(config_name="train",overrides=["model=sonic_latent66"])
    model=instantiate(cfg.model,device="cuda",model_dtype=torch.bfloat16)
    report={"migration":model.initialize_from_checkpoint(args.checkpoint),"gpu":torch.cuda.get_device_name(0)}
    # Small spatial and temporal dimensions keep this an interface test, not a benchmark.
    sample={
        "video":torch.rand(1,3,5,64,64)*2-1,
        "action":torch.randn(1,4,66),
        "proprio":torch.randn(1,4,31),
        "context":torch.randn(1,8,4096),
        "context_mask":torch.ones(1,8,dtype=torch.bool),
        "action_is_pad":torch.tensor([[False,False,False,True]]),
        "action_dim_is_pad":torch.zeros(1,4,66,dtype=torch.bool),
    }
    model.train()
    loss,metrics=model.training_loss(sample)
    if not torch.isfinite(loss):raise RuntimeError("Non-finite loss")
    loss.backward()
    for name,layer in [("action_input",model.action_expert.action_encoder),
                       ("action_output",model.action_expert.head),("state_input",model.proprio_encoder)]:
        grad=layer.weight.grad
        if grad is None or not torch.isfinite(grad).all() or not grad.abs().sum()>0:
            raise RuntimeError("Missing, zero or non-finite gradient: "+name)
    report["training"]={"loss":float(loss.detach()),"metrics":metrics,"backward":"passed",
                        "sample_video_shape":[1,3,5,64,64],"optimizer_steps":0}
    model.zero_grad(set_to_none=True)
    result=model.infer_action(prompt=None,input_image=sample["video"][0,:,0],
        proprio=sample["proprio"][0,0],context=sample["context"],context_mask=sample["context_mask"],
        action_horizon=4,num_inference_steps=2,action_infer_mode="first_frame",seed=42)["action"]
    if result.shape!=(4,66) or not torch.isfinite(result).all():raise RuntimeError("Invalid sampled action")
    report["sampling"]={"shape":list(result.shape),"steps":2,"mode":"first_frame","finite":True}
    report["peak_allocated_GiB"]=torch.cuda.max_memory_allocated()/2**30
    Path(args.report).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__":main()
