import math
import torch
import quarot
import bitblas
from bitblas import auto_detect_nvidia_target
from bitblas.cache import global_operator_cache, get_database_path
# from . import functional
from torchao.ops import rowwise_scaled_linear_cutlass_s4s4_unified, rowwise_scaled_linear_cutlass_s8s4


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
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16, **kwargs):
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
            
        self.de_weight = None
        key = in_features-out_features
        key = f"{in_features}-{out_features}"
        self.a16_matmul = kwargs.get(key, None)
        # if self.a16_matmul is None:
        #     breakpoint()
        self.mask  =(1 << 3) | (1 << 7)
        # breakpoint()

            
    
    def forward(self,x,C=None,**kwargs):
        if kwargs.get("w4a4",False):
            return self.forward_w4a4(x,C)
        elif kwargs.get("w4a8",False):
            return self.forward_w4a8(x,C)
        else:
            return self.forward_w4a16(x)    
        
        
    
    def forward_w4a4(self, x,C=None):
        #if torch.cuda.current_device() != x.device:
        #    torch.cuda.set_device(x.device)
        
        # TODO Improve the following code, move the shape handing to the init.
        assert type(x) == quarot.PackedQuantizedTensor #Quantized input is given
        x, scales_x = x.quantized_x, x.scales_x
        # print(x.shape)
        if self.weight_scales.dim() == 2:
            self.weight_scales = self.weight_scales.view(-1)
        if self.weight.dtype != torch.int8:
            self.weight = self.weight.to(torch.int8)
        if C is None:
            C = torch.empty(x.shape[0], self.weight.shape[0], dtype=torch.float16, device="cuda")
        
        rowwise_scaled_linear_cutlass_s4s4_unified(x, scales_x, self.weight, self.weight_scales, self.bias, C)
        # out = quarot.fuse_matmul(x, scales_x, self.weight, self.weight_scales, self.bias, C)
        return C
    
    def forward_w4a8(self, x,C=None):
        #if torch.cuda.current_device() != x.device:
        #    torch.cuda.set_device(x.device)
        
        # TODO Improve the following code, move the shape handing to the init.
        assert type(x) == quarot.PackedQuantizedTensor # Quantized input is given
        x, scales_x = x.quantized_x, x.scales_x
        # print(x.shape)
        if self.weight_scales.dim() == 2:
            self.weight_scales = self.weight_scales.view(-1)
        if self.weight.dtype != torch.int8:
            self.weight = self.weight.to(torch.int8)
        output = rowwise_scaled_linear_cutlass_s8s4(x, scales_x, self.weight, self.weight_scales, self.bias)
        return output
    

    def forward_w4a16(self, x,C=None):
        if self.weight_scales.dim() == 2:
            self.weight_scales = self.weight_scales.view(-1)
        if self.weight.dtype != torch.int8:
            self.weight = self.weight.to(torch.int8)
            
        if C is None:
            C = torch.empty(x.shape[0], self.weight.shape[0], dtype=torch.float16, device="cuda")
        
        # unpacked_weight = quarot.functional.unpack_i4(self.weight.to(torch.uint8))  # [7168, 5120]
        # unpacked_weight = unpacked_weight.to(torch.float16)
        # if self.weight_scales.dim() == 1:
        # # weight_scales: [7168] -> [7168, 1] for broadcasting
        #     scales = self.weight_scales.view(-1, 1)  # [7168, 1]
        # else:
        #     scales = self.weight_scales
            
        # fp16_weight = unpacked_weight * scales  # [7168, 5120]

        
        self.a16_matmul(x, self.weight ^ self.mask , output=C, scale = self.weight_scales, bias = self.bias)
        
        return C
        
        
        
        

    @staticmethod
    def from_float(module: torch.nn.Linear, weight_scales=None,**kwargs):
        '''
        Generate a new Linear4bit module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        '''
        weight_matrix = module.weight.data

        int_module = Linear4bit(module.in_features, module.out_features, bias=module.bias is not None, dtype=weight_matrix.dtype,**kwargs).to(weight_matrix.dtype)
        if weight_scales is not None:
            assert weight_scales.shape == (module.out_features, 1), 'weight_scales should have shape (out_features, 1)'
            weight_matrix = weight_matrix.cuda()
            int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
            int_rounded_weight = (weight_matrix/weight_scales.cuda()).round()
            # round [-8,7]
            # int_rounded_weight = torch.clamp(int_rounded_weight, -8, 7).to(torch.int8)
            int_module.weight.copy_(quarot.pack_i4(int_rounded_weight.to(torch.int8)).cpu())
        
            if module.bias is not None:
                int_module.bias.copy_(module.bias)
        
        return int_module



def get_matmul_op_w4a16(in_features, out_features, with_scaling = True, enable_tuning = True, fast_decoding = False, bias=False):
    '''
    Get the matmul operator for the 4-bit linear layer.
    '''

    bitblas.logger.setLevel("INFO")
    # from bitblas.ops.impl.matmul_dequantize_impl import (
    #     matmul_nt_dequantize_b,
    #     matmul_nt_dequantize_b_propagate_b,
    #     matmul_nt_dequantize_b_propagate_a_propagate_b,
    # )
    
    
    matmul_config_a16 = bitblas.MatmulConfig(
        M = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096], #, 256*3, 1024, 2048],
        N = out_features,
        K = in_features,
        A_dtype= "float16",
        W_dtype= "int4",
        accum_dtype="float16",
        out_dtype="float16",
        layout="nt",
        with_bias=bias,
        propagate_b=False,
        fast_decoding=fast_decoding,
        group_size=-1,
        with_scaling=with_scaling,
        with_zeros=False,
        zeros_mode=None,
    )
    # matmul_a16 = bitblas.Matmul(config=matmul_config_a16, enable_tuning=False)
    # return matmul_a16     

    config = matmul_config_a16
    
    BITBLAS_TARGET = auto_detect_nvidia_target()
    BITBLAS_DATABASE_PATH = get_database_path()
    
    if global_operator_cache.size() == 0:
        global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
        bitblas.logger.info(f"Loaded {global_operator_cache.size()} operators from database.")

    bitblas_matmul = global_operator_cache.get(config)
    if bitblas_matmul is None:
        # should disable tuning for the first time because we may require loading bitblas operator from database.
        bitblas_matmul = bitblas.Matmul(config, target=BITBLAS_TARGET, enable_tuning=False)
        if enable_tuning:
            bitblas_matmul.hardware_aware_finetune(topk=20)
            global_operator_cache.add(config, bitblas_matmul)
            global_operator_cache.save_into_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
            print("BitBLAS Tuning done, appended operator to global_operator_cache.")
        else:
            print("BitBLAS Operator created.")
    else:
        print("BitBLAS Operator found in global_operator_cache.")
    return bitblas_matmul