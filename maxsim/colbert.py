#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Optional

import faiss
import os
import string
import torch
import torch.nn as nn
import contextlib
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertPreTrainedModel, BertTokenizer
from transformers import AutoTokenizer, PretrainedConfig
from transformers import DistilBertModel, DistilBertPreTrainedModel

class DocumentEncoder:
    def encode(self, texts, **kwargs):
        pass

    @staticmethod
    def _mean_pooling(last_hidden_state, attention_mask):
        token_embeddings = last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class ColBertConfig(PretrainedConfig):
    model_type = "colbert"

    def __init__(self, code_dim=128, **kwargs):
        self.code_dim = code_dim
        super().__init__(**kwargs)


class ColBERT(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.dim = 128
        self.bert = BertModel(config, add_pooling_layer=False)
        self.linear = nn.Linear(config.hidden_size, self.dim, bias=False)
        self.skiplist = dict()
        self.init_weights()

    def use_puct_mask(self, tokenizer):
        encode = lambda x: tokenizer.encode(x, add_special_tokens=False)[0]
        self.skiplist = {w: True
                for symbol in string.punctuation
                for w in [symbol, encode(symbol)]}

    def punct_mask(self, input_ids):
        PAD_CODE = 0
        mask = [
            [(x not in self.skiplist) and (x != PAD_CODE) for x in d]
            for d in input_ids.cpu().tolist()
        ]
        return mask

    def query(self, inputs):
        Q = self.bert(**inputs)[0] # last-layer hidden state
        # Q: (B, Lq, H) -> (B, Lq, dim)
        Q = self.linear(Q)
        # return: (B, Lq, dim) normalized
        lengths = inputs['attention_mask'].sum(1).cpu().numpy()
        return torch.nn.functional.normalize(Q, p=2, dim=2), lengths

    def doc(self, inputs):
        D = self.bert(**inputs)[0]
        D = self.linear(D) # (B, Ld, dim)
        # apply punctuation mask
        if self.skiplist:
            ids = inputs['input_ids']
            mask = torch.tensor(self.punct_mask(ids), device=ids.device)
            D = D * mask.unsqueeze(2).float()
        lengths = inputs['attention_mask'].sum(1).cpu().numpy()
        return torch.nn.functional.normalize(D, p=2, dim=2), lengths

    def score(self, Q, D, mask, in_batch_negs):
        Q = Q.permute(0, 2, 1)
        if in_batch_negs:
            D = D.unsqueeze(1)
            mask = mask.unsqueeze(1)
        # inference: (B, Ld, dim) x (B, dim, Lq) -> (B, Ld, Lq)
        # in-batch negs: (B, 1, Ld, dim) x (B/2, dim, Lq) -> (B, B/2, Ld, Lq)
        cmp_matrix = D @ Q
        # only mask doc dim, query dim will be filled with [MASK]s
        cmp_matrix = cmp_matrix * mask
        best_match = cmp_matrix.max(-2).values # best match per query
        scores = best_match.sum(-1) # sum score over each query
        return scores, cmp_matrix

    def forward(self, Q, D, in_batch_negs=False):
        q_reps, _ = self.query(Q)
        d_reps, _ = self.doc(D)
        d_mask = D['attention_mask'].unsqueeze(-1) # [B, Ld, 1]
        return self.score(q_reps, d_reps, d_mask, in_batch_negs=in_batch_negs)


class ColBertEncoder(DocumentEncoder):
    def __init__(self, model: str, prepend_tok: str, max_ql=32, max_dl=128,
        tokenizer: Optional[str] = None, device: Optional[str] = 'cuda:0',
        query_augment: bool = False, use_puct_mask: bool = True):

        # determine encoder prepend token
        prepend_tokens = ['[Q]', '[D]']
        assert prepend_tok in prepend_tokens
        self.actual_prepend_tokens = ['[unused0]', '[unused1]'] # for compatibility of original Colbert ckpt
        prepend_map = dict(list(zip(prepend_tokens, self.actual_prepend_tokens)))
        self.actual_prepend_tok = prepend_map[prepend_tok]
        self.query_augment = (query_augment and prepend_tok == '[Q]')

        print(f'max_ql={max_ql}, max_dl={max_dl}')

        # load model
        print('Using vanilla ColBERT:', model, tokenizer)
        self.model = ColBERT.from_pretrained(model,
            tie_word_embeddings=True
        )
        self.dim = 128
        self.maxlen = {'[Q]': max_ql, '[D]': max_dl}[prepend_tok]
        self.prepend = True
        # load tokenizer and add special tokens
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer or model)
        print("Original vocab size:", len(self.tokenizer))
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': self.actual_prepend_tokens
        })
        print("New vocab size:", len(self.tokenizer))
        print("Embedding matrix size:", self.model.get_input_embeddings().num_embeddings)
        # print("Token IDs:", self.tokenizer.convert_tokens_to_ids(self.actual_prepend_tokens))
        # print(dir(self.model))
        # print("Original embedding shape:", self.model.bert.embeddings.word_embeddings.weight.shape)
        self.model.resize_token_embeddings(len(self.tokenizer))
        print("Embedding matrix size:", self.model.get_input_embeddings().num_embeddings)
        print("print tokens:", self.tokenizer.convert_ids_to_tokens([101,     2, 30559, 30555, 30531, 30546, 30533, 30555, 30529, 30531,
         30546, 30533, 30555, 30531, 30546, 30535, 30555, 30529, 30531, 30546,
         30535, 30555, 30531, 30546, 30586, 30555, 30529, 30531, 30546, 30586,
         30555,   102]))
        # print("New embedding shape:", self.model.bert.embeddings.word_embeddings.weight.shape)

        if use_puct_mask:
            # mask punctuations
            self.model.use_puct_mask(self.tokenizer)

        # specify device
        self.device = device
        self.model.to(self.device)

    def encode(self, texts, titles=None, return_enc=False,
            fp16=False, sep='\n', debug=False, **kwargs,):
        # preprocess input fields
        prepend_contents = []
        for b, text in enumerate(texts):
            title = titles[b] if titles is not None else None
            content = text if title is None else f'{title}{sep}{text}'
            # prepend special tokens
            content = f'{self.actual_prepend_tok} {content}' if self.prepend else content
            prepend_contents.append(content)

        padding='max_length' if self.query_augment else 'longest'
        # print("padding is: ", padding)
        # tokenize
        enc_tokens = self.tokenizer(prepend_contents, max_length=self.maxlen,
            padding='max_length' if self.query_augment else 'longest',
            truncation=True, return_tensors="pt")
        ids, mask = enc_tokens['input_ids'], enc_tokens['attention_mask']
        # print(ids)
        # query augmentation
        if self.query_augment:
            ids, mask = enc_tokens['input_ids'], enc_tokens['attention_mask']
            ids[ids == 0]   = 103
            #mask[mask == 0] = 1 # following original colbert, no mask change

        enc_tokens.to(self.device)

        if debug:
            for b, ids in enumerate(enc_tokens['input_ids']):
                print(f'--- ColBertEncoder Batch#{b} ---')
                print(self.tokenizer.decode(ids))
                break

        # actual encoding
        if fp16:
            amp_ctx = torch.cuda.amp.autocast()
        else:
            amp_ctx = contextlib.nullcontext()

        with torch.no_grad():
            with amp_ctx:
                if self.actual_prepend_tok == self.actual_prepend_tokens[0]:
                    embs, lengths = self.model.query(enc_tokens)
                else:
                    embs, lengths = self.model.doc(enc_tokens)
                if return_enc:
                    return embs, lengths, enc_tokens
                else:
                    return embs, lengths

from torch import Tensor            
from transformers import BertConfig, BertModel

class Our_Bert(BertModel):
    def __init__(
            self,
            config: BertConfig,
            add_pooling_layer: bool,
            reduce_dim: bool,
            dim: int,
    ) -> None:
        super().__init__(config=config, add_pooling_layer=add_pooling_layer)
        self.reduce_dim = reduce_dim
        if self.reduce_dim:
            self.linear = nn.Linear(
                in_features=config.hidden_size,
                out_features=dim,
                bias=False,
            )

    def forward(self, token_ids: Tensor, attn_mask: Tensor) -> Tensor:
        x = super().forward(
            input_ids=token_ids, attention_mask=attn_mask
        ).last_hidden_state
        if self.reduce_dim:
            x = self.linear(x)

        return x

class ColBertEncoder_math(DocumentEncoder):
    def __init__(self, model_ckpt: str, config_file: str, prepend_tok: str, max_ql=32, max_dl=128,
        tokenizer: Optional[str] = None, device: Optional[str] = 'cuda:0',
        query_augment: bool = False, use_puct_mask: bool = True):

        # determine encoder prepend token
        prepend_tokens = ['[Q]', '[D]']
        assert prepend_tok in prepend_tokens
        self.actual_prepend_tokens = ['[unused0]', '[unused1]'] # for compatibility of original Colbert ckpt
        prepend_map = dict(list(zip(prepend_tokens, self.actual_prepend_tokens)))
        self.actual_prepend_tok = prepend_map[prepend_tok]
        self.query_augment = (query_augment and prepend_tok == '[Q]')

        print(f'max_ql={max_ql}, max_dl={max_dl}')

        # load model
        print('Using vanilla ColBERT:', model_ckpt, tokenizer)
        config = BertConfig.from_json_file(json_file=config_file)

        self.model = Our_Bert(
            config=config,
            add_pooling_layer=False,
            reduce_dim=True,
            dim=128,
        )
        ckpt = torch.load(f=model_ckpt, map_location=device)
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict=ckpt['model_state_dict'], strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        self.dim = 128
        self.maxlen = {'[Q]': max_ql, '[D]': max_dl}[prepend_tok]
        self.prepend = True
        # load tokenizer and add special tokens
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer or model_ckpt)
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': self.actual_prepend_tokens
        })
        self.model.resize_token_embeddings(len(self.tokenizer))
        print(f"Vocab size: {self.tokenizer.vocab_size}")
        print(f"Vocab size: {len(self.tokenizer)}")
        print("Embedding matrix size:", self.model.get_input_embeddings().num_embeddings)
        # print("print tokens:", self.tokenizer.convert_ids_to_tokens([101,  2,   102]))
        # specify device
        self.device = device
        self.model.to(device)

    def encode(self, texts, titles=None, return_enc=False,
            fp16=False, sep='\n', debug=False, pure_math = False, **kwargs,):
        # preprocess input fields
        prepend_contents = []
        for b, text in enumerate(texts):
            title = titles[b] if titles is not None else None
            content = text if title is None else f'{title}{sep}{text}'
            # prepend special tokens
            content = f'{self.actual_prepend_tok} {content}' if self.prepend else content
            prepend_contents.append(content)

        padding='max_length' if self.query_augment else 'longest'
        # print("padding is: ", padding)
        # tokenize
        # enc_tokens = self.tokenizer(prepend_contents, max_length=self.maxlen,
        #     padding='max_length' if self.query_augment else 'longest',
        #     truncation=True, return_tensors="pt")
        enc_tokens = self.tokenizer(prepend_contents, max_length=self.maxlen,
            padding='longest',
            truncation=True, return_tensors="pt")
        ids, mask = enc_tokens['input_ids'], enc_tokens['attention_mask']
        # print(ids)
        # query augmentation
        if self.query_augment:
            ids, mask = enc_tokens['input_ids'], enc_tokens['attention_mask']
            ids[ids == 0]  = 103
            # print(ids)
            #mask[mask == 0] = 1 # following original colbert, no mask change

        enc_tokens.to(self.device)

        if debug:
            for b, ids in enumerate(enc_tokens['input_ids']):
                print(f'--- ColBertEncoder Batch#{b} ---')
                print(self.tokenizer.decode(ids))
                break

        # actual encoding
        if fp16:
            amp_ctx = torch.cuda.amp.autocast()
        else:
            amp_ctx = contextlib.nullcontext()

        def query(inputs, pure_math=False):
            Q = self.model(token_ids=inputs['input_ids'], attn_mask=inputs['attention_mask'])
            # return: (B, Lq, dim) normalized
            lengths = inputs['attention_mask'].sum(1).cpu().numpy()
            return torch.nn.functional.normalize(Q, p=2, dim=2), lengths

        def doc(inputs, pure_math=False):
            D = self.model(token_ids=inputs['input_ids'], attn_mask=inputs['attention_mask'])
            # return: (B, Lq, dim) normalized
            # print(inputs['attention_mask'].shape)
            # assert inputs['attention_mask'].shape[0] == 1
            # length = inputs['attention_mask'][0].sum().item()
            lengths = inputs['attention_mask'].sum(1).cpu().numpy()
            return torch.nn.functional.normalize(D, p=2, dim=2), lengths

        with torch.no_grad():
            with amp_ctx:
                if self.actual_prepend_tok == self.actual_prepend_tokens[0]:
                    embs, lengths = query(enc_tokens, pure_math)
                else:
                    embs, lengths = doc(enc_tokens, pure_math)
                if return_enc:
                    return embs, lengths, enc_tokens
                else:
                    return embs, lengths


