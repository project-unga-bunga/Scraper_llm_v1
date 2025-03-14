import sys

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class LlmScraper(nn.Module):
    def __init__(self,model_name,device):
        super().__init__()
        self.model_name=model_name
        self.device=device
        self.model = (
            AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        ))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prompt_gen(self,prompt:str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=30, eos_token_id=self.tokenizer.eos_token_id)
        response = outputs[0]
        decoded_response = self.tokenizer.decode(response, skip_special_tokens=False)
        return decoded_response

    def create_safe_tensor_model(self)-> dict:
        model_tensor = {}
        for idx in range(1, 4):
            safetensor_path = f"{self.model_name}/model-0000{idx}-of-00003.safetensors"
            with safe_open(safetensor_path, framework="pt", device=0) as f:
                for k in f.keys():
                    model_tensor[k] = f.get_tensor(k)
        return model_tensor