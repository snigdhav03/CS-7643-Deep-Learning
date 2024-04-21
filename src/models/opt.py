import torch.nn as nn
from transformers import OPTForCausalLM, GPT2Tokenizer


class OPT(nn.Module):
    def __init__(self, model_name, device='cpu', sample=False, top_k=None, top_p=None, cache_dir=None):
        super(OPT, self).__init__()
        self.device = device
        self.model = OPTForCausalLM.from_pretrained(f'facebook/{model_name}', cache_dir=cache_dir)
        self.tokenizer = GPT2Tokenizer.from_pretrained(f'facebook/{model_name}', cache_dir=cache_dir)
        self.model.to(self.device)
        self.sample = sample
        self.top_k = top_k
        self.top_p = top_p

    def forward(self, text):
        tokens = self.tokenizer(text, return_tensors='pt', padding=True).to(self.device)
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
