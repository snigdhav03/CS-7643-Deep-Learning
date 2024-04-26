from datetime import datetime

import torch
import torch.nn as nn
from transformers import OPTForCausalLM, GPT2Tokenizer
from src.adapters.addAdapter import add_adapter

from src.models.opt_with_classifier import OPTWithLMClassifier


class OPT(nn.Module):
    def __init__(self, model_name, adapter_name, device='cpu', sample=False, top_k=None, top_p=None, cache_dir=None,
                 mode='generator', adapter_config=None, checkpoint=None):
        super(OPT, self).__init__()
        self.model_name = model_name
        self.adapter_name = adapter_name if adapter_name else 'none'
        self.checkpoint = checkpoint
        self.device = device
        self.mode = mode
        if mode == 'classifier':
            self.model = OPTWithLMClassifier.from_pretrained(checkpoint, cache_dir=cache_dir)
        elif mode == 'generator':
            self.model = OPTForCausalLM.from_pretrained(checkpoint, cache_dir=cache_dir)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        if adapter_name:
            self.model = add_adapter(self.model, adapter_name, adapter_config)
        self.tokenizer = GPT2Tokenizer.from_pretrained(f'facebook/{model_name}', cache_dir=cache_dir)
        self.model.to(self.device)
        self.sample = sample
        self.top_k = top_k
        self.top_p = top_p

    def forward(self, text, labels=None):
        tokens = self.tokenizer(text, return_tensors='pt', padding=True).to(self.device)
        if self.mode == 'classifier':
            return self.classifier_forward(tokens, labels)
        elif self.mode == 'generator':
            out = self.generation_forward(tokens)
            return out, None

    def generation_forward(self, tokens):
        generation_args = {
            'max_new_tokens': 30,
            'do_sample': self.sample,
            'top_k': self.top_k if self.sample else None,
            'top_p': self.top_p if self.sample else None
        }
        generation_args = {k: v for k, v in generation_args.items() if v is not None}
        generated_ids = self.model.generate(**tokens, **generation_args)
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return output

    def classifier_forward(self, tokens, labels):
        labels = torch.tensor([self.tokenizer.encode(label, add_special_tokens=False)[0] for label in labels], device=self.device)
        output = self.model(**tokens, labels=labels)
        logit, loss = output.logits, output.loss
        return logit, loss

    def get_name(self):
        if 'facebook' in self.checkpoint:
            timestamp_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            return f'models--local--{self.model_name}--{self.adapter_name}--{timestamp_str}'
        else:
            return self.checkpoint

    def save(self, path_dir):
        path = f'./{path_dir}/{self.get_name()}'
        self.model.save_pretrained(path)
        return path
