import argparse
import transformers
import torch
import shutil
import json
import numpy as np

from e2e.quantized_qwen import modeling_qwen2
from e2e.checkpoint_utils import data_utils, gptq_utils, rotation_utils,hadamard_utils, quant_utils
from quarot.functional import pack_i4

def main(args):
    transformers.set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    print('Seed set to:', args.seed)
    
   
    # Load model weights from safetensors files
    from safetensors import safe_open
    import json
    
    model_path = '/data/DeepSeek-R1-Distill-Qwen-14B-quarot-w4a4kv4-tp1-with-had-with-quantizer'
    
    # Load the safetensors index to find all shard files
    with open(f'{model_path}/model.safetensors.index.json', 'r') as f:
        index = json.load(f)
    
    old_dict = {}
    # Load all safetensors files
    for filename in index['weight_map'].values():
        if filename not in [f for f in set(index['weight_map'].values())]:
            continue
        filepath = f'{model_path}/{filename}'
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                old_dict[key] = f.get_tensor(key)
    
    # Load quantizers
    quantizers = torch.load(f'{model_path}/new_quantizers.pt', map_location='cpu')
    # remove all the '.module' from the keys
    
      # 🔧 添加权重融合逻辑
    def fuse_qkv_weights(old_dict, quantizers):
        """融合q/k/v权重为qkv_proj"""
        fused_dict = {}
        fused_quantizers = {}
        
        # 获取所有层数
        layers = set()
        for key in old_dict.keys():
            if 'model.layers.' in key and '.self_attn.' in key:
                layer_num = key.split('.')[2]
                layers.add(int(layer_num))
        
        for layer_idx in sorted(layers):
            layer_prefix = f"model.layers.{layer_idx}.self_attn"
            
            # 检查是否存在q/k/v权重
            q_key = f"{layer_prefix}.q_proj.weight"
            k_key = f"{layer_prefix}.k_proj.weight"
            v_key = f"{layer_prefix}.v_proj.weight"
            
            if all(key in old_dict for key in [q_key, k_key, v_key]):
                # 融合权重 [q, k, v] 按行拼接
                q_weight = old_dict[q_key]
                k_weight = old_dict[k_key] 
                v_weight = old_dict[v_key]
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                fused_dict[f"{layer_prefix}.qkv_proj.weight"] = qkv_weight
                
                # 融合bias（如果存在）
                q_bias_key = f"{layer_prefix}.q_proj.bias"
                k_bias_key = f"{layer_prefix}.k_proj.bias"
                v_bias_key = f"{layer_prefix}.v_proj.bias"
                
                if all(key in old_dict for key in [q_bias_key, k_bias_key, v_bias_key]):
                    q_bias = old_dict[q_bias_key]
                    k_bias = old_dict[k_bias_key]
                    v_bias = old_dict[v_bias_key]
                    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                    fused_dict[f"{layer_prefix}.qkv_proj.bias"] = qkv_bias
                
                # 融合quantizers
                q_quant_key = f"{layer_prefix}.q_proj"
                k_quant_key = f"{layer_prefix}.k_proj"
                v_quant_key = f"{layer_prefix}.v_proj"
                
                if all(key in quantizers for key in [q_quant_key, k_quant_key, v_quant_key]):
                    # 假设scale需要拼接
                    q_scale = quantizers[q_quant_key]['scale']
                    k_scale = quantizers[k_quant_key]['scale']
                    v_scale = quantizers[v_quant_key]['scale']
                    
                    # 创建新的quantizer对象
                    qkv_quantizer = quantizers[q_quant_key]  # 复制一个
                    qkv_quantizer['scale'] = torch.cat([q_scale, k_scale, v_scale], dim=0)
                    fused_quantizers[f"{layer_prefix}.qkv_proj"] = qkv_quantizer
                
                print(f"✅ Fused layer {layer_idx} q/k/v -> qkv_proj")
            else:
                # 如果没有完整的q/k/v，保持原样
                for key in [q_key, k_key, v_key]:
                    if key in old_dict:
                        fused_dict[key] = old_dict[key]
        
        return fused_dict, fused_quantizers
    
    def fuse_gate_up_weights(old_dict, quantizers):
        """融合gate/up权重为gate_up_proj"""
        fused_dict = {}
        fused_quantizers = {}
        
        # 获取所有层数
        layers = set()
        for key in old_dict.keys():
            if 'model.layers.' in key and '.mlp.' in key:
                layer_num = key.split('.')[2]
                layers.add(int(layer_num))
        
        for layer_idx in sorted(layers):
            layer_prefix = f"model.layers.{layer_idx}.mlp"
            
            # 检查是否存在gate/up权重
            gate_key = f"{layer_prefix}.gate_proj.weight"
            up_key = f"{layer_prefix}.up_proj.weight"
            
            if all(key in old_dict for key in [gate_key, up_key]):
                # 融合权重 [gate, up] 按行拼接
                gate_weight = old_dict[gate_key]
                up_weight = old_dict[up_key]
                gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)
                fused_dict[f"{layer_prefix}.gate_up_proj.weight"] = gate_up_weight
                
                # 融合quantizers
                gate_quant_key = f"{layer_prefix}.gate_proj"
                up_quant_key = f"{layer_prefix}.up_proj"
                
                if all(key in quantizers for key in [gate_quant_key, up_quant_key]):
                    gate_scale = quantizers[gate_quant_key]['scale']
                    up_scale = quantizers[up_quant_key]['scale']
                    
                    # 创建新的quantizer对象
                    gate_up_quantizer = quantizers[gate_quant_key]  # 复制一个
                    gate_up_quantizer['scale'] = torch.cat([gate_scale, up_scale], dim=0)
                    fused_quantizers[f"{layer_prefix}.gate_up_proj"] = gate_up_quantizer
                
                print(f"✅ Fused layer {layer_idx} gate/up -> gate_up_proj")
            else:
                # 如果没有完整的gate/up，保持原样
                for key in [gate_key, up_key]:
                    if key in old_dict:
                        fused_dict[key] = old_dict[key]
        
        return fused_dict, fused_quantizers
    
    # 🔧 执行融合
    print("🔄 Fusing q/k/v weights...")
    qkv_dict, qkv_quantizers = fuse_qkv_weights(old_dict, quantizers)
    
    print("🔄 Fusing gate/up weights...")
    gate_up_dict, gate_up_quantizers = fuse_gate_up_weights(old_dict, quantizers)
    
    # 🔧 合并所有字典
    # 首先复制所有非融合的权重
    final_dict = {}
    final_quantizers = {}
    
    # 添加非attention和mlp的权重
    for key, value in old_dict.items():
        skip_key = False
        # 跳过已融合的权重
        if '.self_attn.q_proj' in key or '.self_attn.k_proj' in key or '.self_attn.v_proj' in key:
            skip_key = True
        if '.mlp.gate_proj' in key or '.mlp.up_proj' in key:
            skip_key = True
            
        if not skip_key:
            final_dict[key] = value
    
    # 添加融合后的权重
    final_dict.update(qkv_dict)
    final_dict.update(gate_up_dict)
    #
    
    # 添加非融合的quantizers
    for key, value in quantizers.items():
        skip_key = False
        if '.self_attn.q_proj' in key or '.self_attn.k_proj' in key or '.self_attn.v_proj' in key:
            skip_key = True
        if '.mlp.gate_proj' in key or '.mlp.up_proj' in key:
            skip_key = True
            
        if not skip_key:
            final_quantizers[key] = value
    
    # 添加融合后的quantizers
    final_quantizers.update(qkv_quantizers)
    final_quantizers.update(gate_up_quantizers)
    
    print(f"📊 Final dict: {len(final_dict)} weights, {len(final_quantizers)} quantizers")
    
    # 🔧 更新后续处理逻辑
    old_dict = final_dict
    quantizers = final_quantizers
    
    breakpoint()
    

    config = modeling_qwen2.QuarotQwenConfig.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2"
    )
    
    with transformers.modeling_utils.no_init_weights(): 
        new_model = modeling_qwen2.Qwen2QuaRotForCausalLM(vllm_config=config)
    
    breakpoint()
      
    key_maps = {
        "mlp.down_proj": "mlp.down_proj.2",
        # "self_attn.o_proj": "self_attn.o_proj.1"
    }
    bad_key_names = {
        "post_attention_layernorm.weight",
        "input_layernorm.weight",
        "quantizer",
        "module",
        "had",
        "model.norm.weigh"
    }
   
    def _get_new_key(key):
        new_key = key
        for old_name, new_name in key_maps.items():
            new_key = new_key.replace(old_name, new_name)
        return new_key
    
    def _keep_key(key):
        return all(bad_name not in key for bad_name in bad_key_names)

    new_dict = {_get_new_key(key): value for key, value in old_dict.items() if _keep_key(key)}
    breakpoint()
    for key, value in quantizers.items():
        new_key = _get_new_key(key)
        weight_scales = value['scale'].to('cuda')
        new_dict[f"{new_key}.weight_scales"] = weight_scales
        weight_matrix = new_dict[f"{new_key}.weight"].to('cuda')
        int_rounded_weight = (weight_matrix/weight_scales).round().to(torch.int8)
        # keep the int_rounded_weight in the range of [-8, 7]
        int_rounded_weight = torch.clamp(int_rounded_weight, -8, 7)
        new_dict[f"{new_key}.weight"] = pack_i4(int_rounded_weight)

    # config = modeling_llama.QuarotLlamaConfig.from_pretrained(
    #     args.pretraiend_path_or_name,
    #     attn_implementation="flash_attention_2"
    # )
    # torch.set_default_dtype(torch.float16)
    # with transformers.modeling_utils.no_init_weights(): 
    #     new_model = modeling_llama.QuarotLlamaForCausalLM(config=config)

    breakpoint()
    result = new_model.load_state_dict(new_dict, strict=False)
    assert all("had_rem_dim" in key for key in result.missing_keys), result
    assert len(result.unexpected_keys) == 0, result

    new_model = new_model.cpu()

    # new_model.save_pretrained(args.save_path)
    from safetensors.torch import save_file
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 获取模型的state_dict
    state_dict = new_model.state_dict()
    
    # 保存为safetensors格式（推荐）
    save_file(state_dict, os.path.join(args.save_path, "model.safetensors"))
    
    # 或者保存为pytorch格式
    # torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
    
    print(f"✅ Model saved to {args.save_path}")
    with open(f"{model_path}/config.json") as f:
        config = json.load(f)
    config["auto_map"] = {
        "AutoConfig": "QuarotQwenConfig",
        "AutoModelForCausalLM": "Qwen2QuaRotForCausalLM"
    }
    config["model_type"] =  "qwen2_quarot"
    with open(f"{args.save_path}/config.json", "w") as f:
        json.dump(config, f)
    
    shutil.copy("e2e/quantized_qwen/modeling_qwen2.py", f"{args.save_path}/quarot.py")


if __name__ == "__main__":
    import os
    import torch
    import torch.distributed as dist
    
    # 1. 首先初始化分布式环境
    from vllm.distributed import parallel_state
    
    # 设置环境变量（如果还没设置）
    if not dist.is_initialized():
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '12355') 
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        os.environ.setdefault('LOCAL_RANK', '0')
        
        # 2. 初始化分布式环境
        parallel_state.init_distributed_environment(
            world_size=1,
            rank=0, 
            distributed_init_method="env://",
            local_rank=0,
            backend="nccl" if torch.cuda.is_available() else "gloo"
        )
        print("✅ Distributed environment initialized")
    
    # 3. 初始化模型并行组（这是关键步骤）
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,    # tp=1
            pipeline_model_parallel_size=1   # pp=1  
        )
        print("✅ Model parallel groups initialized")
    
    # 现在可以安全使用get_pp_group()了
    pp_group = parallel_state.get_pp_group()
    print(f"Pipeline parallel group: world_size={pp_group.world_size}, is_first_rank={pp_group.is_first_rank}")
    
    parser = argparse.ArgumentParser()

    supported_models = [
        'meta-llama/Llama-2-7b-hf',
        'meta-llama/Llama-2-13b-hf',
        'meta-llama/Llama-2-70b-hf',
    ]

    supported_datasets = ['wikitext2', 'ptb', 'c4']

    # General Arguments
    parser.add_argument('--pretraiend_path_or_name', type=str, default='/data/Meta-Llama-3-8B-Instruct',
                        help='Model to load;')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--eval_dataset', type=str, default='wikitext2',
                        help='Dataset for Evaluation (default: wikitext2)', choices=supported_datasets,)
    

    parser.add_argument('--w_groupsize', type=int, default=-1, 
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--w_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric weight quantization (default: False)')
    parser.add_argument('--w_rtn', action=argparse.BooleanOptionalAction, default=False,
                        help='Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ')
    parser.add_argument('--w_clip', action=argparse.BooleanOptionalAction, default=False,
                        help='''Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization''')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--cal_dataset', type=str, default='wikitext2',
                        help='calibration data samples for GPTQ.', choices=supported_datasets)
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', action=argparse.BooleanOptionalAction, default=False,
                        help='act-order in GPTQ')

    args = parser.parse_args()

    args.w_bits = 4
    main(args)
