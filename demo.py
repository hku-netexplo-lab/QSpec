import sys
sys.path.append('/workspace/qspec/vllm')
import os
os.environ["PYTHONPATH"] = "/workspace/qspec/v1/QuaRot"
from vllm import EngineArgs
from vllm import LLMEngine

import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser
import torch

BG_BLUE = "\033[44m"
BG_GREEN = "\033[42m"
BG_PINK = "\033[45m"
RESET = "\033[0m"


'''
python demo.py --model models/L3 \
    --speculative_model models/L3 \
    --num-speculative-tokens 3 \
    --trust_remote_code --enforce_eager
    
python vllm_test.py --model /workspace/qspec/models/Meta-Llama-3-8B-Instruct \
    --speculative_model /workspace/qspec/models/ \
    --num-speculative-tokens 3 \
    --trust_remote_code --enforce_eager --gpu_memory_utilization 0.3 --max_num_seqs 512
    
python vllm_test.py --model /workspace/qspec/models/Meta-Llama-3-8B-Instruct \
    --num-speculative-tokens 3  --speculative_model "[ngram]" --ngram_prompt_lookup_max 4\
    --trust_remote_code --enforce_eager --max_num_seqs 512
    
                
ncu --set full -o qspec_ncu_sd_mb04 python vllm_test.py --model /workspace/qspec/models/QuaRot/L3  \
    --speculative_model /workspace/qspec/models/QuaRot/L3  \
    --num-speculative-tokens 3   \
    --trust_remote_code --enforce_eager   --max_num_seqs 32
    
    
python vllm_test.py --model /workspace/qspec/models/Meta-Llama-3-8B-Instruct \
    --trust_remote_code --enforce_eager --max_num_seqs 32
    
    
    
    
lm_eval --model vllm --model_args pretrained=/workspace/qspec/models/QuaRot/L3,\
speculative_model=/workspace/qspec/models/QuaRot/L3,num_speculative_tokens=3,\
trust_remote_code=True,enforce_eager=True --tasks tinyGSM8k
                
'''

def create_test_prompts() -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    # get test prompts from dataset-wild-chat
    import datasets
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, stop_token_ids=[128001, 128009], max_tokens=1024, stop=["Question:"])
    # start to load dataset
    dataset = datasets.load_dataset("openai/gsm8k",'main')['train']
    # # save dataset to /workspace/qspec/datasets
    # dataset.save_to_disk("/workspace/qspec/datasets/LongBench")
    # load dataset from /workspace/qspec/datasets
    # dataset = datasets.load_from_disk("/workspace/qspec/datasets/wild-chat")
    # dataset = datasets.load_from_disk("/workspace/qspec/datasets/gsm8k")
    # dataset = dataset["multi_news_e"]

    # # breakpoint()
    shot_num = 10
    prefix = ''
    for i in range(shot_num):
        prefix += 'Question: '+ dataset[i]["question"] + "  Answer: " + dataset[i]["answer"] + '\n'
    
    prompts = []
    i = 0
    len_dataset = len(dataset)-1
    num_prompts = 256
    import random
    from vllm import get_conv_template_name, get_conv_template

    while len(prompts) < min(len_dataset, num_prompts):
        # prompts.append(dataset[i])
        conv_t = get_conv_template_name("Meta-Llama3-8B-Instruct")
        conv = get_conv_template(conv_t)
        # skip unsafe conversations.
        rand_idx = random.randint(0, len_dataset)
        # should_skip = dataset[rand_idx]["toxic"] or dataset[rand_idx]["redacted"]
        # if should_skip:
        #     continue
        # raw_prompt = dataset[rand_idx]["conversation"][0]["content"]
        raw_prompt = prefix + 'Question: ' + dataset[rand_idx]["question"] + " Answer: " 
        # raw_prompt = dataset[rand_idx]["context"]
    

        conv.append_message(conv.roles[0], raw_prompt)
        conv.append_message(conv.roles[1], "")
        prompts.append(conv.get_prompt())
    print(f"{BG_PINK}There are {len(prompts)} prompts to be processed.{RESET}")
    # the average length of the prompt
    avg_len = sum([len(prompt) for prompt in prompts]) / len(prompts)
    print(f"{BG_PINK}The average length of the prompt is {avg_len}.{RESET}")
    return [(prompt, sampling_params) for prompt in prompts]
    
    return [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100)),
        ("To be or not to be,",
         SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100)),
        ("What is the meaning of life?",
         SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100)),

    ]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    finish_count = 0
    tokens_count = 0
    
    import time
    start = time.perf_counter()
    print(f"{BG_BLUE}Start processing requests...{RESET}")

    # start profiling
    # prof = torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("/workspace/qspec/vllm_profiling"),
    #     record_shapes=True,
    #     profile_memory=False,
    #     with_stack=True
    # )
    count_step = 0
    
    
    try:
        while test_prompts or engine.has_unfinished_requests():
            if test_prompts:
                prompt, sampling_params = test_prompts.pop(0)
                engine.add_request(str(request_id), prompt, sampling_params)
                request_id += 1
            # breakpoint()
            request_outputs: List[RequestOutput] = engine.step()
            count_step += 1
            
            # if count_step == 32:
            #     print(f"{BG_GREEN}Start profiling...{RESET}")
            #     prof.start() 

            # if count_step == 32+16:
            #     # stop profiling
            #     print(f"{BG_GREEN}Stop profiling...{RESET}")
            #     prof.stop()
            #     exit(0)
            for request_output in request_outputs:
                if request_output.finished:
                    finish_count += 1
                    tokens_count += len(request_output.outputs[0].token_ids)
                    # print(f"{BG_GREEN}Request {request_output.request_id} finished.{RESET}")
                    # print(f"Prompt: {request_output.prompt}")
                    # print(f"Response: {request_output.outputs[0].text}") # token_ids
                    
    except Exception as e:
        # print the error message
        end = time.perf_counter()
        import traceback
        traceback.print_exc()
        # print e message, reason and traceback
        # breakpoint()
        print(f"{BG_BLUE}Error processing requests, have to stop.{RESET}")
        print(f"{BG_BLUE}Finished processing requests.{RESET}")
        print(f"{BG_BLUE}Time elapsed: {end - start} seconds.{RESET}")
        print(f"{BG_BLUE}Total requests: {request_id}, finished requests: {finish_count}.{RESET}")
        print(f"{BG_BLUE}Total tokens: {tokens_count}.{RESET}")
        print(f"{BG_BLUE}End to end throughput: {tokens_count / (end - start)} tokens per second.{RESET}")
        exit(1)
    
                
    end = time.perf_counter()
    print(f"{BG_BLUE}Finished processing requests.{RESET}")
    print(f"{BG_BLUE}Time elapsed: {end - start} seconds.{RESET}")
    print(f"{BG_BLUE}Total requests: {request_id}, finished requests: {finish_count}.{RESET}")
    print(f"{BG_BLUE}Total tokens: {tokens_count}.{RESET}")
    print(f"{BG_BLUE}End to end throughput: {tokens_count / (end - start)} tokens per second.{RESET}")
    
                
    # end of the engine
                
    


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    test_prompts = create_test_prompts()
    engine = initialize_engine(args)
    process_requests(engine, test_prompts)


if __name__ == '__main__':
    
    # fix all the random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
