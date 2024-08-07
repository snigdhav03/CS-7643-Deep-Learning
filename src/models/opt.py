from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import OPTForCausalLM, GPT2Tokenizer
from src.adapters.addAdapter import add_adapter
from src.const import cache_dir, checkpoint_dir

from src.models.opt_with_classifier import OPTWithLMClassifier
import os


class OPT(nn.Module):
    def __init__(self, model_name, adapter_name, device='cpu', sample=False, top_k=None, top_p=None, cache_dir=None,
                 mode='generator', adapter_config=None, checkpoint=None):
        super(OPT, self).__init__()
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.checkpoint = checkpoint
        self.device = device
        self.mode = mode
        self.adapter_name = adapter_name
        self.adapter_config = adapter_config
        facebook_checkpoint = f'facebook/{model_name}'
        if mode == 'classifier':
            self.model = OPTWithLMClassifier.from_pretrained(facebook_checkpoint, cache_dir=cache_dir)
        elif mode == 'generator':
            self.model = OPTForCausalLM.from_pretrained(facebook_checkpoint, cache_dir=cache_dir)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        if adapter_name and adapter_name is not None:
            self.model = add_adapter(self)
        self.model.to(self.device)
        if checkpoint is not None:    
            path = os.path.join(checkpoint_dir, checkpoint, 'pytorch_model.bin')
            print(f'\n\nLoading Model from checkpoint: {path}\n\n')
            self.load_state_dict(torch.load(path, map_location=self.device))
        self.tokenizer = GPT2Tokenizer.from_pretrained(f'facebook/{model_name}', cache_dir=cache_dir)
        self.sample = sample
        self.top_k = top_k
        self.top_p = top_p

    def forward(self, text, labels=None):
        self.model = self.model.to(self.device)
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
        result = self.get_classification_probabilities(logit)
        result, loss = result.to(self.device), loss.to(self.device)
        return result, loss

    def get_classification_probabilities(self, logit):
        words = ['No', 'Yes']
        label_ids = {word: self.tokenizer.convert_tokens_to_ids(word) for word in words}
        prob = F.softmax(logit, dim=-1)
        mask = torch.full_like(prob, 0)
        for word, idx in label_ids.items():
            mask[:, idx] = 1
        prob = prob * mask
        prob = prob / prob.sum(dim=-1, keepdim=True)
        # new_logits = torch.full_like(logit, -float('inf'))
        # for word, idx in label_ids.items():
        #     new_logits[:, idx] = logit[:, idx]
        # prob = F.softmax(new_logits, dim=-1)
        prob = prob[:, [label_ids[word] for word in words]]
        return prob

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
