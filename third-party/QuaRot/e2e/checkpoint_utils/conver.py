import torch
from safetensors.torch import load_file

# /workspace/qspec/models/QuaRot/True/model.safetensors
# load the safe tensor model
ref_key_dict = load_file("/workspace/qspec/models/QuaRot/True/model.safetensors")

# /workspace/qspec/models/QuaRot/Llama-2-7b-hf-4b-rotate-new
# load the fp16 weight and quantizers

save_dict = torch.load("/workspace/qspec/models/QuaRot/Llama-2-7b-hf-4b-rotate-new")
quantizers = save_dict["w_quantizers"]
key_maps = {
    "mlp.down_proj": "mlp.down_proj.2",
    "self_attn.o_proj": "self_attn.o_proj.1"
}


    