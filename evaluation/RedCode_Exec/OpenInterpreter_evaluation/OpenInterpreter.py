from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from interpreter import interpreter
import re
import os

class BaseModel:
    def __init__(self, model_name, temperature, top_p, seed, max_tokens, dry_run, device: str='cuda'):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_tokens = max_tokens
        self.dry_run = dry_run
        self.device = device

        if 'claude' in model_name:
            # anthropic
            raise NotImplementedError()
            import anthropic
            api_key = os.environ['ANTHROPIC_API_KEY']
            self.client = anthropic.Anthropic(api_key=api_key)
        elif 'gpt' in model_name:
            # openai
            raise NotImplementedError()
            if model_name == 'gpt-4o':
                api_key = os.environ['OPENAI_PROJ_API_KEY']
                self.client = OpenAI(api_key=api_key)
            elif model_name == 'gpt-35-turbo':
                self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
            else:
                self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        else:
            # assume local model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True, 
                device_map='auto',
                torch_dtype=torch.bfloat16
            ).to(device)