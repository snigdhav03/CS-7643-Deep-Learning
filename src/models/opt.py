import torch
import torch.nn as nn
from transformers import OPTForCausalLM, AutoTokenizer
from src.const import cache_dir


class OPT(nn.Module):
    def __init__(self, model_name, device='cpu', sample=False, top_k=None, top_p=None):
        super(OPT, self).__init__()
        self._device = device
        self._model_name = model_name
        # Casual LM for all? I guess.
        self._model = OPTForCausalLM.from_pretrained(f'facebook/{model_name}', cache_dir=cache_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(f'facebook/{model_name}', cache_dir=cache_dir, padding_side='left')
        self._model.to(self._device)
        self._sample = sample
        self._top_k = top_k
        self._top_p = top_p

    def forward(self, x):
        tokens = self._tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        tokens = tokens.to(self._device)
        if not self._sample:
            generated_ids = self._model.generate(**tokens, max_new_tokens=30, do_sample=self._sample)
        else:
            generated_ids = self._model.generate(**tokens, max_new_tokens=30, do_sample=self._sample, top_k=self._top_k, top_p=self._top_p)
        return self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

