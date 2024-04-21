# models/model_manager.py
from transformers import pipeline, LlamaTokenizer, LlamaForCausalLM
import torch

class ModelManager:
    def __init__(self, model_name: str, local_path=None, batch_size=4):
        if local_path:  # Path to local model directory is provided
            # if "jamshid" in model_name.lower():
            #     self.tokenizer = LlamaTokenizer.from_pretrained(local_path)
            #     self.model = LlamaForCausalLM.from_pretrained(local_path)
            #     self.pipeline = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)
            # else:
            #     self.pipeline = pipeline('text-generation', model=local_path, torch_dtype=torch.float16)
            if model_name == "llama-2-7b-chat-hf":
                # self.tokenizer = LlamaTokenizer.from_pretrained(local_path)
                # self.model = LlamaForCausalLM.from_pretrained(local_path)
                # self.pipeline = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)
                self.pipeline = pipeline('text-generation', model=local_path, torch_dtype=torch.float16)
            elif model_name == "gemma-7b":
                self.pipeline = pipeline('text-generation', model=local_path, torch_dtype=torch.float16)
            elif model_name == "gemma-2b":
                self.pipeline = pipeline('text-generation', model=local_path, torch_dtype=torch.bfloat16, device=0)
            elif model_name == "mistral-7b-instruct":
                self.pipeline = pipeline('text-generation', model=local_path, torch_dtype=torch.float16)
            else:
                self.pipeline = pipeline('text-generation', model=local_path, torch_dtype=torch.bfloat16, device=0, batch_size=batch_size)
                self.pipeline.tokenizer.pad_token_id = self.pipeline.model.config.eos_token_id
        else:
            self.pipeline = pipeline('text-generation', model=model_name, torch_dtype=torch.bfloat16)
        self.model = self.pipeline.model

    # def infer(self, text: str):
    #     return self.pipeline(text, max_length=50)

    def infer(self, texts):
        # Assuming 'texts' is a list of strings
        return self.pipeline(texts, truncation=True)
    
    def calculate_model_size(self):
        """ Calculate the memory usage of the model based on its parameters. """
        total_size = 0
        for param in self.model.parameters():
            total_size += param.data.numel() * param.data.element_size()  # numel() gives total number of elements, element_size() gives size in bytes
        return total_size / (1024 ** 2)  # Convert bytes to megabytes
    def calculate_model_size2(self):
        """Calculate the memory usage of the model's parameters and buffers."""
        total_size = 0
        for param in self.model.parameters():
            total_size += param.numel() * param.element_size()
        for buffer in self.model.buffers():
            total_size += buffer.numel() * buffer.element_size()
        return total_size / (1024 ** 2)  # Convert bytes to megabyte
