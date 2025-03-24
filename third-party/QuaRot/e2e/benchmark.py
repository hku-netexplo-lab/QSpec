import argparse
import gc
import pprint
import numpy as np
import torch
import time

from e2e.quantized_llama import modeling_llama
import torch
import transformers
from checkpoint_utils import rotation_utils
from transformers import AutoTokenizer

model_configs = [
    #"meta-llama/Llama-2-7b-hf",
    # "meta-llama/Llama-2-13b-hf", 
    # "meta-llama/Llama-2-70b-hf", 
    #"../../../models/QuaRot/True",
    "/workspace/qspec/models/Llama-2-7b-hf",
]

tokenizer_path = '/workspace/qspec/models/Meta-Llama-3-8B-Instruct'

benchmark_dtypes = ["int4", torch.float16]
num_warmup_steps = 0
num_bench_steps = 1
# define global variables for the benchmark 'count'
count = 0

def repeated_run(num_repeats=10):
    def func(module):
        def _f(*args, **kwargs):
            times = []
            for i in range(num_repeats):
                times.append(module(*args, **kwargs))
            return tuple(zip(*times))
        return _f
    return func

def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()

@repeated_run()
def module_benchmark(module):
    # warmup
    for i in range(num_warmup_steps):
        out = module()
    torch.cuda.synchronize()
    
    _cleanup()
    torch.cuda.reset_max_memory_allocated()
    start_time = time.perf_counter()
    
    
    for i in range(num_bench_steps):
        out = module()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()

    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps, peak_memory


def get_model_quantized(config_name):
    config_name = "/workspace/qspec/models/QuaRot/L3"
    config = transformers.AutoConfig.from_pretrained(
        config_name,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    dtype_old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)

    with transformers.modeling_utils.no_init_weights(): 
        model = modeling_llama.QuarotLlamaForCausalLM(config=config)
        
        
    
    from safetensors.torch import load_file
    weight_path = "/workspace/qspec/models/QuaRot/L3/model-00001-of-00002.safetensors"
    weight_path2 = "/workspace/qspec/models/QuaRot/L3/model-00002-of-00002.safetensors"
    state_dict = load_file(weight_path)
    state_dict2 = load_file(weight_path2)
    # merge the two state_dicts into one
    state_dict = {**state_dict, **state_dict2}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    torch.set_default_dtype(dtype_old)
    model.load_state_dict(state_dict)
    model.cuda()    
    model.fuse_qkv()
    
    input_text = """
Once upon a time in a land far, far away, there was a small village nestled between rolling hills and a dense, enchanted forest. The villagers lived simple, happy lives, but they were always curious about the mysteries that lay beyond the forest's edge. One day, a young and adventurous villager named Alex decided to venture into the forest to uncover its secrets. As Alex stepped into the forest, the trees seemed to whisper ancient tales, and the air was filled with the scent of magic. What Alex discovered on this journey would change the village forever...

Please continue the story.
"""
    input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()
    output_ids = model.generate(input_ids, max_length=512, num_return_sequences=1)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)

    breakpoint()
    
    from checkpoint_utils import data_utils, eval_utils
    testloader = data_utils.get_loaders(
        name='wikitext2',
        seed=0,
        model="/workspace/qspec/models/Llama-2-7b-hf",
        seqlen=2048,
        eval_mode=True
    )
    #hf_token=args.hf_token,
    model.seqlen = 2048
    dataset_ppl = eval_utils.evaluator(model, testloader,torch.device('cuda:0') , args)
    print(dataset_ppl)
    breakpoint()
    return model


def get_model_hf(config_name):
    return transformers.LlamaForCausalLM.from_pretrained(
        config_name, 
        torch_dtype=torch.float16, 
        attn_implementation="flash_attention_2"
    )
    
def get_model_qserve(config_name):
    from transformers import AutoConfig
    from qserve import LlamaForCausalLM,SamplingParams,EngineArgs,LLMEngine
#     --model '/workspace/qspec/models/Llama-2-7B-QServe' \
#   --benchmarking \
#   --precision w4a8kv4 \
#   --group-size -1
    args = argparse.Namespace(
        model='/workspace/qspec/models/Llama-2-7B-QServe',
        benchmarking=True,
        precision='w4a8kv4',
        group_size=-1
    )
    engine_args = EngineArgs.from_cli_args(args)
    llmEngine = LLMEngine.from_engine_args(engine_args)
    prompt_len = 1024
    batch_size = 16
    
    sp = SamplingParams()
    config = AutoConfig.from_pretrained(
        config_name,
        trust_remote_code=True,
    )
    kv_config = {"INT4_ENABLED": True, "ZEROS_ENABLED": False}
    model = LlamaForCausalLM(config=config,group_size=-1, sampling_params=sp,kv_cache_config=kv_config)
    breakpoint()
    return model
    
    

def get_model_fp16(config_name):
    return modeling_llama.QuarotFP16LlamaForCausalLM.from_pretrained(
        config_name, 
        torch_dtype=torch.float16, 
        attn_implementation="flash_attention_2"
    )


def run_prefill(model, bsz, prefill_length):
    device = model.device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    return module_benchmark(lambda: model(test_input))


def run_decode(model, bsz, prefill_length, decode_steps):
    device = model.device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    model._expected_max_length = prefill_length + decode_steps
    out = model(test_input)
    past_key_values = out.past_key_values
    del out
    _cleanup()
    next_input = torch.tensor([[100] for _ in range (bsz)], dtype=torch.int32, device=device)
    def _decode_for_multiple_steps():
        # if past_key_values not tuple
        if not isinstance(past_key_values, tuple):
            past_key_values.length = prefill_length
        for _ in range(decode_steps):
            model(next_input, past_key_values=past_key_values)
    return module_benchmark(_decode_for_multiple_steps)
    

def run_e2e(model, bsz, prefill_length, decode_steps):
    device = model.device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    next_input = torch.tensor([[100] for _ in range (bsz)], dtype=torch.int32, device=device)
    def _prefill_and_decode_for_multiple_steps():

        # print("Start profiling")
        # prof1 = torch.profiler.profile(
        # activities=[
        #     torch.profiler.ProfilerActivity.CPU,
        #     torch.profiler.ProfilerActivity.CUDA,
        # ],
        # record_shapes=True,
        # profile_memory=False,
        # with_stack=True,
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_quantized'),
        # )
        # prof1.start()
        model._expected_max_length = prefill_length + decode_steps
        out = model(test_input)
        for _ in range(decode_steps):
            model(next_input, past_key_values=out.past_key_values)
        # prof1.stop()
        # print("Stop profiling")
        
    return module_benchmark(_prefill_and_decode_for_multiple_steps)


def _wait_for_input():
    print("Press enter")
    input()

@torch.no_grad
def run_all_for_model(model, bsz, prefill, decode):
    model.eval()
    model = model.cuda()
    time_prefill, _ = run_prefill(model, bsz, prefill)
    _cleanup()
    if decode is not None:
        time_decode, memory_decode = run_decode(model, bsz, prefill, decode)
        _cleanup()
        time_e2e, _ = run_e2e(model, bsz, prefill, decode)
        _cleanup()
    else:
        time_decode = time_e2e = None
    return time_prefill, time_decode, time_e2e, memory_decode

def benchmark(args):
    # prof1 = torch.profiler.profile(
    # activities=[
    #     torch.profiler.ProfilerActivity.CPU,
    #     torch.profiler.ProfilerActivity.CUDA,
    # ],
    # record_shapes=True,
    # profile_memory=False,
    # with_stack=True,
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_quantized'),
    # )
    
    # prof2 = torch.profiler.profile(
    # activities=[
    #     torch.profiler.ProfilerActivity.CPU,
    #     torch.profiler.ProfilerActivity.CUDA,
    # ],
    # record_shapes=True,
    # profile_memory=False,
    # with_stack=True,
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_fp16'),
    # )

    
    
    for config_name in model_configs:
        
        # get_model_qserve(config_name)
        

        model = get_model_quantized(config_name)
        # prof1.start()
        time_prefill_i4, time_decode_i4, time_e2e_i4, mem_i4 = run_all_for_model(
            model, args.batch_size, args.prefill_seq_len, args.decode_steps)
        # prof1.stop()
        del model
        _cleanup()
        # exit()
        

        model = get_model_hf(config_name)
        # prof2.start()
        time_prefill_f16, time_decode_f16, time_e2e_f16, mem_f16 = run_all_for_model(
            model, args.batch_size, args.prefill_seq_len, args.decode_steps)
        # prof2.stop()
        del model
        _cleanup()
       

        print(f"Prefill Int4 time: {np.mean(time_prefill_i4):.3f} +- {1.96 * np.std(time_prefill_i4):.3f}ms")
        print(f"Prefill FP16 time: {np.mean(time_prefill_f16):.3f} +- {1.96 * np.std(time_prefill_f16):.3f}ms")
        print(f"Speedup: {np.mean(time_prefill_f16) / np.mean(time_prefill_i4):.3f}x")
        print(f'Prefill & {config_name} & {args.batch_size} & {args.prefill_seq_len} & {np.mean(time_prefill_f16):.3f} & {np.mean(time_prefill_i4):.3f}\\\\')

        if args.decode_steps is not None:
            print(f"Decode Int4 time: {np.mean(time_decode_i4):.3f} +- {1.96 * np.std(time_decode_i4):.3f}ms")
            print(f"Decode FP16 time: {np.mean(time_decode_f16):.3f} +- {1.96 * np.std(time_decode_f16):.3f}ms")
            print(f"Speedup: {np.mean(time_decode_f16) / np.mean(time_decode_i4):.3f}x")
            print(f'Decode & {config_name} & {args.batch_size} & {args.prefill_seq_len} & {args.decode_steps} & {np.mean(time_decode_f16):.3f} & {np.mean(time_decode_i4):.3f}\\\\')

            print(f"E2E Int4 time: {np.mean(time_e2e_i4):.3f} +- {1.96 * np.std(time_e2e_i4):.3f}ms")
            print(f"E2E FP16 time: {np.mean(time_e2e_f16):.3f} +- {1.96 * np.std(time_e2e_f16):.3f}ms")
            print(f"Speedup: {np.mean(time_e2e_f16) / np.mean(time_e2e_i4):.3f}x")
            print(f'E2E & {config_name} & {args.batch_size} & {args.prefill_seq_len} & {args.decode_steps} & {np.mean(time_e2e_f16):.3f} & {np.mean(time_e2e_i4):.3f}\\\\')
        
        # table-style output

        print(f"Int4 memory: {np.mean(mem_i4) / (1024 * 1024 * 1024):.3f}GB +- {1.96 * np.std(mem_i4):.3f}")
        print(f"FP16 memory: {np.mean(mem_f16) / (1024 * 1024 * 1024):.3f}GB +- {1.96 * np.std(mem_f16):.3f}")
        print(f"Memory saving: {np.mean(mem_f16) / np.mean(mem_i4):.3f}x")
        print(f'Memory saving & {config_name} & {args.batch_size} & {args.prefill_seq_len} & {args.decode_steps} & {np.mean(mem_i4) / (1024 * 1024 * 1024):.3f}GB & {np.mean(mem_f16) / (1024 * 1024 * 1024):.3f}GB\\\\')
        
        print('--------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size', type=int,
        help='Batch size',
        default=16,
    )
    parser.add_argument(
        '--prefill_seq_len', type=int,
        help='Size of the input sequence',
        default=1024,
    )
    parser.add_argument(
        '--decode_steps', type=int,
        help='Decode steps',
        required=False,
        default=200,
    )
    
    # fix all the random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    transformers.set_seed(0)

    args = parser.parse_args()
    pprint.pprint(vars(args))
    benchmark(args)
    

