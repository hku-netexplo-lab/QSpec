import sys
import os
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
1. Download the QSpec model from Huggingface hub copy the path to the model.
2. Users can use LmEval to evaluate the model on downstream tasks. 
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID lm_eval --model vllm --model_args pretrained=PATH-TO-QSPEC-MODEL,\
speculative_model=PATH-TO-QSPEC-MODEL,num_speculative_tokens=3,\
trust_remote_code=True,enforce_eager=True --tasks tinyGSM8k --trust_remote_code
```
3. Users can use demo.py to check the throughput of QSpec on their own machine.
```bash
# QSpec
CUDA_DEVICE_ORDER=PCI_BUS_ID python demo.py --model PATH-TO-QSPEC-MODEL  --speculative_model PATH-TO-QSPEC-MODEL(Same as the former)      --num-speculative-tokens 3   --max_num_seqs 4  --trust_remote_code --enforce_eager
CUDA_DEVICE_ORDER=PCI_BUS_ID python demo.py --model PATH-TO-QSPEC-MODEL  --max_num_seqs 4  --trust_remote_code --enforce_eager 
# Auto-regressive W4A16 without QSpec (Baseline)
```
4. Users can try other counterparts like EAGLE or N-gram etc. by changing the model name in the above commands.
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID python demo.py --model PATH-TO-QSPEC-MODEL \
    --speculative_model PATH-TO-EAGLE \
    --num-speculative-tokens 3 \
    --trust_remote_code --enforce_eager
```             
'''

def create_test_prompts() -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    """You can modify this function to create your own test prompts."""
    """Return a list of tuples, each containing a prompt and its sampling parameters."""
    # get test prompts from dataset-wild-chat
    import datasets
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, stop_token_ids=[128001, 128009], max_tokens=1024, stop=["Question:"])
    # start to load dataset
    dataset = datasets.load_dataset("openai/gsm8k", "main", split="train") # gsm8k
    # dataset = datasets.load_dataset("allenai/WildChat")["train"] # wildchat
    # dataset = datasets.load_dataset("Muennighoff/mbpp", "full", split="test") # mbpp
    # dataset = datasets.load_dataset("philschmid/mt-bench", split="train") # mt-bench
    # dataset = datasets.load_dataset("hendrydong/gpqa_diamond",split="test") # diamond
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    shot_num = 5
    prefix = ''
    for i in range(shot_num):
        prefix += 'Question: '+ dataset[i]["question"] + "  Answer: " + dataset[i]["answer"] + '\n' # gsm8k
        # prefix += 'Question: '+ dataset[i]["text"] + "  Answer: " + dataset[i]["code"] + '\n' # mbpp
        # prefix += 'Question: '+ dataset[i]["problem"] + "  Answer: " + dataset[i]["solution"] + '\n' # diamond

    
    prompts = []
    i = 0
    len_dataset = len(dataset)-1
    num_prompts = 128
    
    import random
    from vllm import get_conv_template_name, get_conv_template
    random.seed(0)
    while len(prompts) < min(len_dataset, num_prompts):
        # prompts.append(dataset[i])
        conv_t = get_conv_template_name("Meta-Llama3-8B-Instruct")
        conv = get_conv_template(conv_t)
        rand_idx = random.randint(0, len_dataset)

        raw_prompt = prefix + 'Question: ' + dataset[rand_idx]["question"] + " Answer: "  # gsm8k
        # raw_prompt = dataset[rand_idx]["context"] # wildchat
        # raw_prompt = prefix + 'Question: ' + dataset[rand_idx]["text"] + " Answer: " # mbpp
        # raw_prompt = prefix + 'Question: ' + dataset[rand_idx]["turns"][0] + " Answer: " # mt-bench
        # raw_prompt = prefix + 'Question: ' + dataset[rand_idx]["problem"] + " Answer: " # diamond

        conv.append_message(conv.roles[0], raw_prompt)
        conv.append_message(conv.roles[1], "")
        prompts.append(conv.get_prompt())
    print(f"{BG_PINK}There are {len(prompts)} prompts to be processed.{RESET}")
    # the average length of the prompt
    avg_len = sum([len(prompt) for prompt in prompts]) / len(prompts)
    print(f"{BG_PINK}The average length of the prompt is {avg_len}.{RESET}")
    return [(prompt, sampling_params) for prompt in prompts]
    
    ### EAMPLE PROMPTS ###
    # return [
    #     ("A robot may not injure a human being",
    #      SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100)),
    #     ("To be or not to be,",
    #      SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100)),
    #     ("What is the meaning of life?",
    #      SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100)),

    # ]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    finish_count = 0
    tokens_count = 0
    
    import time
    start = time.perf_counter()
    print(f"{BG_BLUE}Start processing requests...{RESET}")
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
            
            for request_output in request_outputs:
                if request_output.finished:
                    finish_count += 1
                    tokens_count += len(request_output.outputs[0].token_ids)

                    
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
    exit(0)
                
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
    import multiprocessing as mp
    mp.set_start_method("spawn")
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
