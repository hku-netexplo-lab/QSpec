import math
import torch
import quarot
from torchao.ops import rowwise_scaled_linear_cutlass_s4s4
#import fast_hadamard_transform



class ShapeHandler:
    def __init__(self, x: torch.Tensor):
        self.size_excl_last = x.numel()//x.shape[-1]
        self.shape_excl_last = tuple(x.shape[:-1])

    # Keep the last dim unchanged, flatten all previous dims
    def flatten(self, x: torch.Tensor):
        return x.view(self.size_excl_last, -1)

    # Recover back to the original shape.
    def unflatten(self, x: torch.Tensor):
        return x.view(self.shape_excl_last + (-1,))

    def unflatten_scale(self, x: torch.Tensor):
        return x.view(self.shape_excl_last)


class Linear4bit(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        '''
        Symmetric 4-bit Linear Layer.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_scales',
                             torch.zeros((self.out_features, 1), requires_grad=False,dtype=dtype))
        self.register_buffer('weight', (torch.randint(1, 7, (self.out_features, self.in_features // 2),
                                                             # SubByte weight
                                                             dtype=torch.uint8, requires_grad=False)))
        if bias:                                                        
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None
            
        self.bsz = 1
        self.output_buffer = torch.empty(self.bsz, self.out_features, dtype=torch.float16, device="cuda")
        
    def forward(self, x):
        #if torch.cuda.current_device() != x.device:
        #    torch.cuda.set_device(x.device)
        
        # TODO Improve the following code, move the shape handing to the init.
        assert type(x) == quarot.PackedQuantizedTensor #Quantized input is given
        x, scales_x = x.quantized_x, x.scales_x
        
        if self.weight_scales.dim() == 2:
            self.weight_scales = self.weight_scales.view(-1)
        if self.weight.dtype != torch.int8:
            self.weight = self.weight.to(torch.int8)
            
        C = torch.empty(x.shape[0], self.weight.shape[0], dtype=torch.float16, device="cuda")
        out = quarot.fuse_matmul(x, scales_x, self.weight, self.weight_scales, self.bias, C)
        return out
        #breakpoint()
        if x.shape[0] > 16:
            # breakpoint()
            C = torch.empty(x.shape[0], self.weight.shape[0], dtype=torch.float16, device="cuda")
            out = quarot.fuse_matmul(x, scales_x, self.weight, self.weight_scales, self.bias, C)
        else:
            out = quarot.fuse_matmul(x, scales_x, self.weight, self.weight_scales, self.bias, self.output_buffer)
        return out
        
        
        # #shape_handler = ShapeHandler(quantized_x)
        # #quantized_x = shape_handler.flatten(quantized_x)
        # #x = quarot.matmul(x, self.weight)
        # x = quarot.opt_matmul(x, self.weight)
        
        # #out = shape_handler.unflatten(
        # #    quarot.sym_dequant(int_result, scales_x, self.weight_scales))
        
        # if self.bias is not None:
        #     return quarot.sym_dequant(x, scales_x, self.weight_scales) + self.bias
        # else:
        #     return quarot.sym_dequant(x, scales_x, self.weight_scales)
        
        

    @staticmethod
    def from_float(module: torch.nn.Linear, weight_scales=None,):
        '''
        Generate a new Linear4bit module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        '''
        weight_matrix = module.weight.data
        
        
        int_module = Linear4bit(module.in_features, module.out_features, bias=module.bias is not None, dtype=weight_matrix.dtype).to(weight_matrix.dtype)
        if weight_scales is not None:
            assert weight_scales.shape == (module.out_features, 1), 'weight_scales should have shape (out_features, 1)'
            weight_matrix = weight_matrix.cuda()
            int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
            int_rounded_weight = (weight_matrix/weight_scales.cuda()).round()
            # round [-8,7]
            # int_rounded_weight = torch.clamp(int_rounded_weight, -8, 7).to(torch.int8)
            int_module.weight.copy_(quarot.functional.pack_i4(int_rounded_weight.to(torch.int8)).cpu())
        
            if module.bias is not None:
                int_module.bias.copy_(module.bias)
        
        return int_module
