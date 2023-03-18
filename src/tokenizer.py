import numpy as np
from transformers import AutoTokenizer
from utils import pad_and_truncate, mask_pad


class ABSATokenizer:

    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def text_to_ids(self, text):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        return sequence

    def pad_trunc(self, input_ids, max_seq_len, attention: bool = False):
        input_ids = np.stack(list(map(lambda id: pad_and_truncate(id, max_seq_len), input_ids)))
        if attention:
            attention_mask = np.stack(list(map(lambda id: mask_pad(id), input_ids.copy())))
            return input_ids, attention_mask
        else:
            return input_ids