import torch
from .quantization import pack_i4, unpack_i4, asym_quant_dequant, sym_quant_dequant
from .hadamard import(
    matmul_hadU_cuda, 
    random_hadamard_matrix, 
    apply_exact_had_to_linear,
    opt_matmul_hadU_cuda)


class PackedQuantizedTensor:
    def __init__(self, 
                 quantized_x: torch.Tensor, 
                 scales_x: torch.Tensor):
        self.quantized_x = quantized_x
        self.scales_x = scales_x

    def size(self):
        return self.quantized_x.size()
    
    @property
    def device(self):
        return self.quantized_x.device
    
    @property
    def dtype(self):
        return self.quantized_x.dtype
