import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from wbwam.models.wan22.action_dit import ActionDiT
from wbwam.models.wan22.wan_video_dit import WanVideoDiT
from wbwam.models.wan22.mot import MoT
from wbwam.models.wan22.sonic_latent import WBWAMSonicLatent, CONTRACT, INTERFACE_KEYS
from wbwam.models.wan22.wbwam_optional_idm import WBWAMOptionalIDM

def tiny(cls=WBWAMSonicLatent, state_dim=31, action_dim=66):
    video=WanVideoDiT(hidden_dim=32,in_dim=4,out_dim=4,ffn_dim=64,freq_dim=16,text_dim=32,
        num_heads=2,attn_head_dim=16,num_layers=1,eps=1e-6,patch_size=(1,2,2),
        has_image_input=False,seperated_timestep=True,fuse_vae_embedding_in_latents=True)
    action=ActionDiT(hidden_dim=32,action_dim=action_dim,ffn_dim=64,text_dim=32,freq_dim=16,
        eps=1e-6,num_heads=2,attn_head_dim=16,num_layers=1)
    model=cls(video_expert=video,action_expert=action,mot=MoT(mixtures={"video":video,"action":action}),
        vae=nn.Identity(),proprio_dim=state_dim,text_dim=32,device="cpu",torch_dtype=torch.float32)
    model.action_idm_prob=0.5
    return model

class SonicTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/"physical.pt"
        self.source=tiny(WBWAMOptionalIDM,96,96)
        self.source.save_checkpoint(self.path,step=20560)
        self.model=tiny()
    def tearDown(self): self.tmp.cleanup()
    def test_migration_and_roundtrip(self):
        report=self.model.initialize_from_checkpoint(self.path)
        self.assertEqual(report["training_step"],0)
        src=self.source.mot.state_dict()
        for k,v in self.model.mot.state_dict().items():
            if k not in INTERFACE_KEYS:self.assertTrue(torch.equal(v,src[k]),k)
        other=tiny(); other.initialize_from_checkpoint(self.path)
        self.assertTrue(torch.equal(other.proprio_encoder.weight,self.model.proprio_encoder.weight))
        self.assertEqual(tuple(self.model.action_expert.head.weight.shape),(66,32))
        p=Path(self.tmp.name)/"sonic.pt";self.model.save_checkpoint(p,step=1)
        other.load_checkpoint(p)
        for k,v in self.model.state_dict().items():self.assertTrue(torch.equal(v,other.state_dict()[k]),k)
        payload=torch.load(p,weights_only=True);payload["sonic_contract"]["action_dim"]=96;torch.save(payload,p)
        with self.assertRaises(ValueError):other.load_checkpoint(p)
        with self.assertRaises(RuntimeError):other.infer_action()
    def test_fail_closed(self):
        with self.assertRaises(RuntimeError): self.model.training_loss({})
        d=torch.load(self.path,weights_only=True)
        d["mot"].pop(next(k for k in d["mot"] if k not in INTERFACE_KEYS))
        torch.save(d,self.path)
        with self.assertRaises(ValueError):self.model.initialize_from_checkpoint(self.path)
        self.assertFalse(self.model._sonic_ready)
    def test_masked_loss(self):
        p=torch.ones(1,3,66,requires_grad=True);t=torch.zeros_like(p)
        mask=torch.tensor([[False,False,True]])
        value=self.model._compute_action_loss_per_sample(p,t,mask,None)
        self.assertEqual(value.item(),1)
        t[:,2]=999
        self.assertEqual(self.model._compute_action_loss_per_sample(p,t,mask,None).item(),1)
        value.sum().backward();self.assertEqual(p.grad[:,2].abs().sum().item(),0)
        self.assertEqual(self.model._compute_action_loss_per_sample(p,t,torch.ones_like(mask),None).item(),0)
        self.assertEqual(self.model._compute_action_loss_per_sample(p,t,None,torch.ones_like(p,dtype=torch.bool)).item(),0)
        # Official loss behavior remains unchanged.
        self.assertGreater(self.source._compute_action_loss_per_sample(p,t,mask,None).item(),1)
    def test_action_sampling_and_dimensions(self):
        self.model.initialize_from_checkpoint(self.path)
        args=dict(prompt=None,input_image=None,action_horizon=4,proprio=torch.zeros(31),
            context=torch.randn(1,3,32),context_mask=torch.ones(1,3,dtype=torch.bool),
            num_inference_steps=2,action_infer_mode="no_video",seed=42)
        result=self.model.infer_action(**args)["action"]
        self.assertEqual(tuple(result.shape),(4,66));self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result[:,:64].shape[-1],64);self.assertEqual(result[:,64:].shape[-1],2)
        for dim in [34,96]:
            with self.assertRaises(ValueError):self.model.infer_action(**{**args,"proprio":torch.zeros(dim)})
        with self.assertRaises(ValueError):self.model._compute_action_loss_per_sample(torch.zeros(1,2,96),torch.zeros(1,2,96),None,None)
    def test_configs(self):
        root=Path(__file__).resolve().parents[1]/"configs"
        with initialize_config_dir(config_dir=str(root),version_base=None):
            cfg=compose(config_name="train",overrides=["wb_task=sonic_latent66"])
            self.assertEqual(cfg.model.proprio_dim,31)
            self.assertEqual(cfg.model.action_dit_config.action_dim,66)
            self.assertFalse(cfg.fk_link_loss.enabled)
        with initialize_config_dir(config_dir=str(root),version_base=None):
            cfg=compose(config_name="train",overrides=["model=wbwam_optional_idm"])
            self.assertEqual(cfg.model.proprio_dim,96)
            self.assertEqual(cfg.model.action_dit_config.action_dim,96)

if __name__=="__main__":unittest.main()
