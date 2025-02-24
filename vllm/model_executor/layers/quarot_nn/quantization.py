import quarot
import torch

class Quantizer(torch.nn.Module):
    def __init__(self, input_clip_ratio=1.0,w4a4 = False):
        super().__init__()
        self.input_clip_ratio = input_clip_ratio
        self.w4a4 = w4a4
    
    def forward(self, x):
        # scales_x = (torch.max(torch.abs(x), dim=-1)[0].unsqueeze(1)/7).to(torch.float16) * self.input_clip_ratio
        # quantized_x = quarot.sym_quant(x, scales_x)
        if self.w4a4:
            quantized_x, scales_x = quarot.fuse_sym_quant(x, self.input_clip_ratio)
            packed_tensor = quarot.PackedQuantizedTensor(quantized_x, scales_x)
            return packed_tensor
        else:
            return x
