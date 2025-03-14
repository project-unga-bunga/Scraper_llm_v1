import gc
import time
from dataclasses import asdict

from safetensors import safe_open
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Bielik-7B-v0.1"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# Note : Manual safetensor dict construction
# model_tensor = {}
# for idx in range(1, 4):
#     safetensor_path = f"{model_name}/model-0000{idx}-of-00003.safetensors"
#     with safe_open(safetensor_path, framework="pt", device=0) as f:
#         for k in f.keys():
#             model_tensor[k] = f.get_tensor(k)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    local_files_only=True,
    #load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()
# NOte: Git is pain in the ass
tokenizer = AutoTokenizer.from_pretrained(model_name)

# missing_keys, unexpected_keys = model.load_state_dict(model_tensor, strict=False)
# print("Missing keys:", missing_keys)
# print("Unexpected keys:", unexpected_keys)
# del model_tensor
# gc.collect()
# time.sleep(0.1)

# model.to(device)
prompt = "Hej, czy możesz ze mną pogadać?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=30,eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
