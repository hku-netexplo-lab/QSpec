
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = '/data/Meta-Llama-3-8B-Instruct'
quant_path = '/data/models/Meta-Llama-3-8B-Instruct-awq-128-zero-c4'
quant_config = { "q_group_size": 128, "w_bit": 4, "zero_point": True, "version": "gemm" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')