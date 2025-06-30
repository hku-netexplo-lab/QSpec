import quarot
import torch
# from . import funtional
class Quantizer(torch.nn.Module):
    def __init__(self, input_clip_ratio=1.0,**kwargs):
        super().__init__()
        self.input_clip_ratio = input_clip_ratio
        self.w4a4 = kwargs.get("w4a4", False)
    
    def forward(self, x, scale=None, out=None, **kwargs):
        # scales_x = (torch.max(torch.abs(x), dim=-1)[0].unsqueeze(1)/7).to(torch.float16) * self.input_clip_ratio
        # quantized_x = quarot.sym_quant(x, scales_x)
        if kwargs.get("w4a4",False):
            if scale is None or out is None:
                quantized_x, scales_x = quarot.fuse_sym_quant(x, self.input_clip_ratio)
            else:
                quantized_x, scales_x = quarot.fuse_sym_quant_with_buffer(x,scale,out,self.input_clip_ratio)
            packed_tensor = quarot.PackedQuantizedTensor(quantized_x, scales_x)
            return packed_tensor
        elif kwargs.get("w4a8",False):
            if scale is None or out is None:
                quantized_x, scales_x = quarot.fuse_sym_quant_i8(x, self.input_clip_ratio)
            else:
                quantized_x, scales_x = quarot.fuse_sym_quant_i8_with_buffer(x,scale,out,self.input_clip_ratio)
            packed_tensor = quarot.PackedQuantizedTensor(quantized_x, scales_x)
            return packed_tensor
        else:
            # breakpoint()
            return x
        
