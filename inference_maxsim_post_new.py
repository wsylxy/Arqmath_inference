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
import glob, pickle
import os
import gc
import csv
from tqdm import tqdm

def arqmath_encoder(tok_ckpoint, model_ckpoint, mold, gpu_dev='cuda:0'):
    from maxsim.colbert import ColBertEncoder
    from maxsim.preprocess import preprocess_for_transformer

    max_ql = 128
    max_dl = 512

    colbert_encoder = ColBertEncoder(model_ckpoint,
        '[D]' if mold == 'D' else '[Q]',
        max_ql=max_ql, max_dl=max_dl,
        tokenizer=tok_ckpoint, device=gpu_dev,
        query_augment=True, use_puct_mask=True
    )

    def encoder(batch_psg, debug=False, return_enc=False):
        batch = [preprocess_for_transformer(p) for p in batch_psg]
        file_exists = os.path.isfile("data/test.csv")
        with open("data/test.csv", "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['raw', 'processed'])
            for i in range(len(batch)):
                writer.writerow([batch_psg[i], batch[i]])
        print("raw batch:", batch_psg)  
        print("preprocessed batch:", batch)
        return colbert_encoder.encode(batch,
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

    post_cnt = embedding_maxsim_post(    #uncomment this for embs generation
        xml_file_path=cfg.DATA.XML_FILE,
        emb_filepate=cfg.DATA.EMBS_FILE,
        encoder=doc_encoder
    )   #emb_pooling: num of formulas in pool x emb_size

    # ########################## load and compuate query embs #############################

    def extract_number(filename):
        match = re.search(r'emb_(\d+)\.pt', filename)
        return int(match.group(1)) if match else float('inf')

    # def extract_number(path):
    #     return int(os.path.basename(path).split(".")[-2])  # "word_emb.0.pt" -> 0
    
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
            # qcode = qcode.squeeze(dim=0)
            query_embs[qid] = qcode.to(torch.float16)
            # print(qcode.dtype)
        batch_sz=400
        emb_files = sorted(glob.glob(os.path.join(cfg.DATA.EMBS_FILE, "emb_*.pt")), key=extract_number)
        post_cnt = 0
        for path in emb_files:
            f = torch.load(path, map_location='cpu')
            doc_ids = f['doc_ids']
            post_cnt += len(doc_ids)
        print("post_cnt: ", post_cnt)
        total_sim = {query_id: torch.empty(post_cnt, device=DEVICE) for query_id in query_embs}
        doc_base_idx = 0
        pool_id = []
        for we_file in tqdm(emb_files, desc="loading embeddings"):
        # for we_file, id_file, len_file in tqdm(zip(emb_files, doc_ids_files, doc_len_files), total=len(emb_files), desc="Loading embeddings"):
            print(1)
            one_emb_file = torch.load(we_file, map_location='cpu')
            flat_embs_all = one_emb_file['embs']         # [sum(L_i), D]
            doc_lens_all = one_emb_file['lengths']       # list of L_i
            doc_ids_all = one_emb_file['doc_ids']        # list of doc_id
            # with open(id_file, 'rb') as f:
            #     doc_ids_all = pickle.load(f)
            # with open(len_file, 'rb') as f:
            #     doc_lens_all = pickle.load(f)
            print("flat_embs_all.shape[0]==", flat_embs_all.shape[0])
            print("sum(doc_lens_all)==", sum(doc_lens_all))
            print("len(doc_ids_all)==", len(doc_ids_all))
            print("len(doc_lens_all)==", len(doc_lens_all))

            assert flat_embs_all.shape[0] == sum(doc_lens_all) and len(doc_ids_all) == len(doc_lens_all)
            D = flat_embs_all.size(-1)
            offset = 0
            for i in range(0, len(doc_lens_all), batch_sz):
                doc_lens = doc_lens_all[i:i + batch_sz]
                doc_ids = doc_ids_all[i:i + batch_sz]
                B = len(doc_lens)
                max_len = max(doc_lens)
                batch_embs = torch.zeros(B, max_len, D, dtype=flat_embs_all.dtype)
                for j in range(B):
                    L = doc_lens[j]
                    if offset + L > flat_embs_all.size(0):  # 防止越界
                        break
                    batch_embs[j, :L] = flat_embs_all[offset:offset + L]
                    offset += L
                word_offsets = torch.arange(max_len).unsqueeze(0).expand(B, max_len)
                doc_lens_tensor = torch.tensor(doc_lens).unsqueeze(1)
                mask = (word_offsets < doc_lens_tensor).to(DEVICE)  # [B, L]
                # print("batch_embs.shape: ", batch_embs.shape)
                batch_embs = batch_embs.to(DEVICE)
                pool_id.extend(doc_ids)
                for query_id, query_emb in query_embs.items():
                    # print("query_emb.T.shape:", query_emb.T.shape)
                    sim = batch_embs @ query_emb.permute(0, 2, 1) #[B, Ld, D] x [1, D, Lq] = [B, Ld, Lq]
                    sim = sim.permute(0, 2, 1)  #[B, Lq, Ld]
                    sim = sim * mask.unsqueeze(1)   #[B, Lq, Ld]
                    sim = sim.float()
                    max_values = torch.max(input=sim, dim=-1).values   #[B, Lq]
                    score = max_values.sum(dim=1)   #[B]
                    total_sim[query_id][doc_base_idx:doc_base_idx+B] = score
                    del query_emb
                doc_base_idx += B  
            
            # for batch_embs, batch_lens, batch_ids in zip(one_emb_file['embs'], one_emb_file['lengths'], one_emb_file["doc_ids"]):
            #     # batch_embs: [B, L, D]
            #     # batch_lens: List[int] of length B
            #     B, L, D = batch_embs.shape
            #     pool_id.extend(batch_ids)
            #     #build attention mask
            #     word_offsets = torch.arange(L).unsqueeze(0).expand(B, L)  # [B, L]
            #     doc_lens_tensor = torch.tensor(batch_lens).unsqueeze(1)   # [B, 1]
            #     mask = (word_offsets < doc_lens_tensor).to(DEVICE)  # [B, L]
            #     for query_id, query_emb in query_embs.items():
            #         sim = batch_embs @ query_emb.permute(0, 2, 1) #[B, Ld, D] x [D, Lq] = [B, Ld, Lq]
            #         sim = sim.permute(0, 2, 1)  #[B, Lq, Ld]
            #         sim = sim * mask.unsqueeze(1)   #[B, Lq, Ld]
            #         sim = sim.float()
            #         max_values = torch.max(input=sim, dim=-1).values   #[B, Lq]
            #         score = max_values.sum(dim=1)   #[B]
            #         total_sim[query_id][doc_base_idx:doc_base_idx+B] = score
            #         del query_emb
            #     doc_base_idx += B
            
            print(f"GPU mem: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        # print(doc_base_idx)
        # print(total_sim)
        torch.save(total_sim, cfg.DATA.SIM)
        print("length of pool_id is:", len(pool_id))
        print("final doc_idx is:", doc_base_idx)
        assert len(pool_id) == doc_base_idx
        
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
        
