import quarot
import torch
# from . import functional


class OnlineHadamard(torch.nn.Module):
    def __init__(self, hadamard_dim, force_fp32=False):
        super().__init__()
        self.fp32_had = force_fp32
        # hadamard_dim: 32(num of head) for attn,  intermediate size for mlp
        had_rem_dim, self.rem_dim = quarot.functional.hadamard.get_hadK(hadamard_dim)
        self.scale_32 = 1.0/torch.tensor(32).sqrt()
        self.scale_14336 = 1.0/torch.tensor(14336).sqrt()
        if had_rem_dim is not None:
            self.register_buffer("had_rem_dim", had_rem_dim)
            if not self.fp32_had:
                self.had_rem_dim = self.had_rem_dim.to(torch.float16).cuda()
        else:
            self.had_rem_dim = None
            
        self.qspec = False       
    
    def forward(self, x, out = None, **kwargs):
        if kwargs.get("w4a4", False):
            self.qspec = True
            
        x_dtype = x.dtype
        if self.fp32_had:
            x = x.float()
        if self.rem_dim==1:
            if self.qspec:
                return quarot.functional.opt_matmul_hadU_cuda(x, self.had_rem_dim, self.rem_dim,out = out,scale = self.scale_32)
            else:
                return quarot.functional.matmul_hadU_cuda(x, self.had_rem_dim, self.rem_dim,out = out,scale = self.scale_32)

        if self.qspec:
            x = quarot.functional.opt_matmul_hadU_cuda(x, self.had_rem_dim, self.rem_dim,out = out,scale = self.scale_14336)
        else:
            x = quarot.functional.matmul_hadU_cuda(x, self.had_rem_dim, self.rem_dim,out = out,scale = self.scale_14336)
        x = x.to(x_dtype)
        return x
