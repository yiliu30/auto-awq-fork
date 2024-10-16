from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import transformers
import torch

from torchutils import freeze_seed

freeze_seed()
limit = 20
from torchutils.eval import eval_wikitext2

model_path = "mistralai/Mistral-7B-Instruct-v0.2"
quant_path = "mistral-instruct-v0.2-awq"
model_path = "Qwen/Qwen1.5-0.5B"
quant_path = "qwen1.5-0.5b-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

load_model_options = {"low_cpu_mem_usage": True, "use_cache": False, "torch_dtype": torch.float32}

device = torch.device("cuda")
float_model = transformers.AutoModelForCausalLM.from_pretrained(model_path, **load_model_options)
float_model.eval()
float_model.to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)


result_float_model = eval_wikitext2(float_model, tokenizer, limit=limit)
print(f"Float model perplexity: {result_float_model}")

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path, **load_model_options)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.eval()
result_awq_float_model = eval_wikitext2(model.to(device), tokenizer, limit=limit)
print(f"AWQ float model perplexity: {result_awq_float_model}")
# Quantize
model.quantize(tokenizer, quant_config=quant_config)
result_awq_model = eval_wikitext2(model.to(device), tokenizer, limit=limit)

print(f"AWQ model perplexity: {result_awq_model}")
# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')


model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=False)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=False)
result_awq_reload_qmodel = eval_wikitext2(model.to(device), tokenizer, limit=limit)
print(f"AWQ reloaded model perplexity: {result_awq_reload_qmodel}")

"""
{'perplexity': 15.3238, 'prediction_time': 0.131}
Float model perplexity: {'perplexity': 15.3238, 'prediction_time': 0.131}

{'perplexity': 15.3238, 'prediction_time': 0.072}
AWQ float model perplexity: {'perplexity': 15.3238, 'prediction_time': 0.072}

{'perplexity': 16.7228, 'prediction_time': 0.055}
AWQ model perplexity: {'perplexity': 16.7228, 'prediction_time': 0.055}

perplexity 16.7228
time 0.148  sec
{'perplexity': 16.7228, 'prediction_time': 0.148}
AWQ reloaded model perplexity: {'perplexity': 16.7228, 'prediction_time': 0.148}

"""
