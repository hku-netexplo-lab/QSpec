import argparse
import transformers
import torch
import shutil
import json
import numpy as np


from e2e.quantized_llama import modeling_llama
from e2e.checkpoint_utils import data_utils, gptq_utils, rotation_utils
from quarot.functional import pack_i4
import hadamard_utils, quant_utils

def main(args):
    transformers.set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    print('Seed set to:', args.seed)
    
    # model = transformers.LlamaForCausalLM.from_pretrained(args.pretraiend_path_or_name, torch_dtype='auto')
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # model.seqlen = 2048
    
    # rotation_utils.fuse_layer_norms(model)
    # rotation_utils.rotate_model(model)
    
    # quant_utils.add_actquant(model) #Add Activation Wrapper to the model
    # qlayers = quant_utils.find_qlayers(model)
    # for name in qlayers:
    #     if 'down_proj' in name:
    #         had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
    #         qlayers[name].online_full_had = True
    #         qlayers[name].had_K = had_K
    #         qlayers[name].K = K
    #         qlayers[name].fp32_had = False
    #     if 'o_proj' in name:
    #         had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
    #         qlayers[name].online_partial_had = True
    #         qlayers[name].had_K = had_K
    #         qlayers[name].K = K
    #         qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
    #         qlayers[name].fp32_had = False


    
    # trainloader = data_utils.get_loaders(
    #     'wikitext2', nsamples=128,
    #     seed=0, model=args.pretraiend_path_or_name,
    #     seqlen=model.seqlen, eval_mode=False
    # )
    # quantizers = gptq_utils.gptq_fwrd(model, trainloader, device, args)


    # save the quantizers first 
    # torch.save(quantizers, f"{args.save_path}/quantizers.pt")
    

    # import quant_utils
    save_dict = torch.load('../../../models/QuaRot/Llama-3-8b-hf-4b-rotate')
    old_dict = save_dict["model"]
    quantizers_dict = save_dict["w_quantizers"]
    # remove all the '.module' from the keys
    quantizers = {key.replace('.module', ''): value for key, value in quantizers_dict.items()}
    
  
    
    # remove model.norm.weight
    
    # breakpoint()
    
    #rplace the keyname 'had_k' with had_rem_dim in old_dict
    # for key in old_dict.keys():
    #     if 'had_K' in key:
    #         new_key = key.replace('had_K', 'had_rem_dim')
    #         old_dict[new_key] = old_dict.pop(key)
    
      
    key_maps = {
        "mlp.down_proj": "mlp.down_proj.2",
        "self_attn.o_proj": "self_attn.o_proj.1"
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
    for key, value in quantizers.items():
        new_key = _get_new_key(key)
        weight_scales = value.scale.to('cuda')
        new_dict[f"{new_key}.weight_scales"] = weight_scales
        weight_matrix = new_dict[f"{new_key}.weight"].to('cuda')
        int_rounded_weight = (weight_matrix/weight_scales).round().to(torch.int8)
        # keep the int_rounded_weight in the range of [-8, 7]
        int_rounded_weight = torch.clamp(int_rounded_weight, -8, 7)
        new_dict[f"{new_key}.weight"] = pack_i4(int_rounded_weight)

    config = modeling_llama.QuarotLlamaConfig.from_pretrained(
        args.pretraiend_path_or_name,
        attn_implementation="flash_attention_2"
    )
    torch.set_default_dtype(torch.float16)
    with transformers.modeling_utils.no_init_weights(): 
        new_model = modeling_llama.QuarotLlamaForCausalLM(config=config)

    breakpoint()
    result = new_model.load_state_dict(new_dict, strict=False)
    assert all("had_rem_dim" in key for key in result.missing_keys), result
    assert len(result.unexpected_keys) == 0, result

    new_model = new_model.cpu()

    new_model.save_pretrained(args.save_path)
    with open(f"{args.save_path}/config.json") as f:
        config = json.load(f)
    config["auto_map"] = {
        "AutoConfig": "quarot.LlamaConfig",
        "AutoModelForCausalLM": "quarot.QuarotLlamaForCausalLM"
    }
    config["model_type"] =  "llama_quarot"
    with open(f"{args.save_path}/config.json", "w") as f:
        json.dump(config, f)
    
    shutil.copy("e2e/quantized_llama/modeling_llama.py", f"{args.save_path}/quarot.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    supported_models = [
        'meta-llama/Llama-2-7b-hf',
        'meta-llama/Llama-2-13b-hf',
        'meta-llama/Llama-2-70b-hf',
    ]

    supported_datasets = ['wikitext2', 'ptb', 'c4']

    # General Arguments
    parser.add_argument('--pretraiend_path_or_name', type=str, default='/workspace/qspec/models/Meta-Llama-3-8B-Instruct',
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
