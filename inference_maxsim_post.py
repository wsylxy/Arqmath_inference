from transformers import BertModel, BertConfig, BertTokenizer
from config import get_config, DEVICE, SEED
from torch import Tensor
from dataset import TOPIC, POOLING
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from transformer import Transformer
from collections import defaultdict
from emb import embedding_maxsim_post
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import torch.nn as nn
import torch.nn.functional as F
import torch
import logger
import pandas as pd
import re
import glob
import os
import gc
from tqdm import tqdm

def arqmath_encoder(tok_ckpoint, model_ckpoint, mold, gpu_dev='cuda:0'):
    from maxsim.colbert import ColBertEncoder

    max_ql = 128
    max_dl = 512

    colbert_encoder = ColBertEncoder(model_ckpoint,
        '[D]' if mold == 'D' else '[Q]',
        max_ql=max_ql, max_dl=max_dl,
        tokenizer=tok_ckpoint, device=gpu_dev,
        query_augment=True, use_puct_mask=True
    )

    def encoder(batch_psg, debug=False, return_enc=False):
        # batch_psg = [preprocess_for_transformer(p) for p in batch_psg]
        return colbert_encoder.encode(batch_psg,
            fp16=True, debug=debug, return_enc=return_enc)

    return encoder, (None, colbert_encoder, colbert_encoder.dim)

def gen_flat_topics(collection, kw_sep):
    from maxsim.gen_topic import gen_topics_queries
    from maxsim.preprocess import tokenize_query
    for qid, query, _ in gen_topics_queries(collection):
        # skip topic file header / comments
        if qid is None or query is None or len(query) == 0:
            continue
        # query example: [{'type': 'tex', 'str': '-0.026838601\\ldots'}]
        if len(query) == 1 and query[0]['type'] == 'term':
            query = query[0]['str']
        else:
            query = tokenize_query(query)
            if kw_sep == 'comma':
                query = ', '.join(query)
            elif kw_sep == 'space':
                query = ' '.join(query)
            else:
                raise NotImplementedError
        yield qid, query

def main() -> None:
    cfg = get_config(args=None)

    doc_encoder, (tokenizer, model, dim) = arqmath_encoder(
                            tok_ckpoint=cfg.CKPT.BERT.TOK,
                            model_ckpoint=cfg.CKPT.COLBERT_MODEL,
                            mold='D',  
                        )

    pool_id, post_cnt = embedding_maxsim_post(    #uncomment this for embs generation
        file_path = cfg.DATA.DICT_FILE,
        emb_filepate=cfg.DATA.EMBS_FILE,
        encoder=doc_encoder
    )   #emb_pooling: num of formulas in pool x emb_size

    def extract_number(filename):
        match = re.search(r'emb_(\d+)\.pt', filename)
        return int(match.group(1)) if match else float('inf')
    
    ########################## load and compuate query embs #############################
    query_embs = {}
    with torch.no_grad():
        collection = cfg.DATA.TOPIC
        kw_sep = 'space'
        adhoc_query = None
        topics = gen_flat_topics(collection, kw_sep) if adhoc_query is None else [('adhoc_query', adhoc_query)]
        encoder, enc_utils = arqmath_encoder(
                            tok_ckpoint=cfg.CKPT.BERT.TOK,
                            model_ckpoint=cfg.CKPT.COLBERT_MODEL,
                            mold='Q',  
                        )
        for qid, query in topics:
            qcode, lengths = encoder([query])
            qcode = qcode.squeeze(dim=0)
            # print(qcode.shape)
            # query_embs[qid] = qcode.to(torch.float16)
            query_embs[qid] = qcode

        emb_files = sorted(glob.glob(os.path.join(cfg.DATA.EMBS_FILE, "emb_*.pt")), key=extract_number)
        # print(emb_files)
        # total_sim = defaultdict(list)
        total_sim = {query_id: torch.empty(post_cnt, device=DEVICE) for query_id in query_embs}
        # print(total_sim)
        # print(total_sim['A.301'].shape)
        doc_base_idx = 0
        for path in tqdm(emb_files, desc="loading embeddings"):
            print(1)
            embs_pooling = torch.load(path, map_location='cpu') #list of tensors, each tensor is [L', D], which is the tensor of one pool doc
            count = 1
            for i, doc_emb in enumerate(embs_pooling):
                doc_emb = doc_emb.squeeze(dim=0).to(DEVICE)
                # print('doc_emb', doc_emb.shape)
                for query_id, query_emb in query_embs.items():
                    # print(query_emb)
                    sim = query_emb @ doc_emb.T #[L, L'], L is seqlen of query, L' is seqlen of document
                    # print('1', sim.shape)

                    sim = torch.max(input=sim, dim=-1).values 
                    # print('2', sim.shape)
                    #seqlen = sim.shape[0]
                    sim = sim.sum(dim=-1).detach().squeeze() #scalar
                    doc_idx = doc_base_idx + i
                    # print(sim.shape, sim.device, sim.requires_grad)
                    # total_sim[query_id].append(sim)  #when loop is done, total_sim[query_id] is the score list of pool docs
                    total_sim[query_id][doc_idx] = sim
                    # print(sim)
                    del query_emb
                # print(count)
                count += 1
                del doc_emb
                torch.cuda.empty_cache()

            doc_base_idx += len(embs_pooling)
            del embs_pooling
            # torch.cuda.empty_cache()
            gc.collect()
            print(f"GPU mem: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        # print(doc_base_idx)
        # print(total_sim)
        torch.save(total_sim, cfg.DATA.SIM)
        print("length of pool_id is:", len(pool_id))
        print("final doc_idx is:", doc_idx+1)
        assert len(pool_id) == doc_idx + 1
        
    total_sim = torch.load(cfg.DATA.SIM, map_location=DEVICE)
    total_topk = {}
    for query_id, sim_tensor in total_sim.items():  # sim_tensor 是 GPU tensor [1300000]
        topk_sim, topk_idx = torch.topk(sim_tensor, k=cfg.INF.TOPK)

        # 如果 pool_id 是 list，可以提前转成 tensor
        # pool_id_tensor = torch.tensor(pool_id, device=topk_idx.device)

        # real_topk_idx = pool_id_tensor[topk_idx]  # ✅ GPU 上索引，无需 .tolist()

        real_topk_idx = [pool_id[i] for i in topk_idx.tolist()]

        total_topk[query_id] = {
            "sim": topk_sim,
            "post_idx": real_topk_idx
        }
    # print(total_topk)
    print(2)

    rows = []
    for query_id, data in total_topk.items():
        sims = data["sim"]
        doc_ids = data["post_idx"]
        if isinstance(sims, torch.Tensor):
            sims = sims.detach().cpu().tolist()
        for rank, (doc_id, sim) in enumerate(zip(doc_ids, sims), start=1):
            rows.append({
                "Query_Id": query_id,
                "Post_Id": doc_id,
                "Rank": rank,
                "Score": sim,
                "Run_Number": "Run_0" 
            })
        # rows.extend([(query_id, doc_id, rank, sim, "Run_0") for rank, (doc_id, sim) in enumerate(zip(doc_ids, sims), start=1)])
    df = pd.DataFrame(rows)
    df.to_csv(cfg.DATA.PRED, index=False, sep='\t')

if __name__ == '__main__':
    main()     
        
