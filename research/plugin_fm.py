"""Harness FM-contract wrapper (like siderius_task/plugins/_fm_metrics.SHIM) around a SIDERIUS model plugin.
Env: PFM_PLUGIN (plugin .py), PFM_CKPT (state_dict; empty = random init), PFM_SEED."""
import importlib.util, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "siderius_task", "plugins"))
from _fm_buffer import encode_batch
_spec = importlib.util.spec_from_file_location("cand", os.environ["PFM_PLUGIN"]); _m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
_CFG = json.loads(os.environ.get("FMT_CFG", "{}"))   # size overrides, as in r3_joint.py
EMBED_DIM = int(_m.PLUGIN_MODEL_CLASS(_m.PLUGIN_CONFIG_CLASS(**_CFG)).embed_dim)


class _Enc(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, x):
        return self.model.encode(x)


def to_tensors(dataset, inputs):
    return (torch.from_numpy(encode_batch(dataset, inputs)),)


def load_encoder(dataset, pretrained=True):
    torch.manual_seed(int(os.environ.get("PFM_SEED", "0")))
    model = _m.PLUGIN_MODEL_CLASS(_m.PLUGIN_CONFIG_CLASS(**_CFG))
    ck = os.environ.get("PFM_CKPT", "")
    if pretrained and ck:
        model.load_state_dict(torch.load(ck, map_location="cpu"))
    return _Enc(model)
