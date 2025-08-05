from transformers import BertModel, BertConfig, BertTokenizer
from config import get_config, DEVICE, SEED
from torch import Tensor
from dataset import TOPIC, POOLING
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from transformer import Transformer
from collections import defaultdict
from emb import embedding_maxsim_post_old
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

def main() -> None:
    cfg = get_config(args=None)
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.CKPT.BERT.TOK
    )
    print(0)
    config = BertConfig.from_json_file(cfg.CKPT.BERT.CFG)
    model = BertModel(config, add_pooling_layer=False)

    pool_id, post_cnt = embedding_maxsim_post_old(    #uncomment this for embs generation
        model=model,
        config=config,
        device=DEVICE,
        file_path = cfg.DATA.DICT_FILE,
        ckpt_filepath=cfg.CKPT.BERT.PT,
        tokenizer=tokenizer,
        emb_filepate=cfg.DATA.EMBS_FILE,
    )   #emb_pooling: num of formulas in pool x emb_size
    model.to(device=DEVICE)
    linear = nn.Linear(config.hidden_size, 128, bias=False).to(device=DEVICE)
    model.eval()
    state_dict = torch.load(cfg.CKPT.BERT.PT, map_location="cuda")
    model.load_state_dict(state_dict=state_dict["model_state_dict"], strict=False)
    logger.log_info(f"Loaded model '{cfg.CKPT.BERT.PT}'")
    ################# load query docs#######################
    posts = []
    file = open(file=cfg.DATA.TOPIC_FILE, mode='r', encoding='utf-8')
    for line in file:
        line = line.strip().split(sep='\t')
        posts.append(line)
    file.close()
    query_docs = {}  #{"topic_id": post_i}
    for line in posts:
        query_id = line[1]
        query_docs[query_id] = line[0]

    def extract_number(filename):
        match = re.search(r'emb_(\d+)\.pt', filename)
        return int(match.group(1)) if match else float('inf')
    
    ########################## load and compuate query embs #############################
    query_embs = {}
    with torch.no_grad():
        for query_id, query_post in tqdm(query_docs.items(), desc="loading topics"):
            # src = [tokenizer.encode(expr=expr) for expr in query_post]
            tokens = tokenizer(
                text=query_post,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=cfg.MODEL.BERT.MAX_SEQ_LEN,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            input_ids = tokens["input_ids"].to(device=DEVICE)
            attn_mask = tokens["attention_mask"].to(device=DEVICE, dtype=torch.bool)
            # print(attn_mask)
            query_emb = model(input_ids=input_ids, attention_mask=attn_mask)    #[B, L, D] embedding of formulas under the current query doc
            query_emb = query_emb.last_hidden_state
            
            query_emb = linear(query_emb)  
            # attn_mask = attn_mask.squeeze(dim=(-3, -2))
            pad_mask = attn_mask == 0
            n_pad = pad_mask.int().sum(dim=-1)
            eoe_ids = attn_mask.size(dim=-1) - n_pad - 1
            batch_ids = torch.arange(
                start=0, end=attn_mask.size(dim=0), dtype=torch.int64, device=attn_mask.device
            )
            attn_mask[batch_ids, eoe_ids] = False
            attn_mask[:, 0] = False
            query_emb = query_emb * (attn_mask).unsqueeze(dim=-1).float()
            # print(query_emb)
            query_emb = query_emb.squeeze(dim=0)
            query_emb = F.normalize(input=query_emb, p=2.0, dim=-1, eps=1e-12)
            query_embs[query_id] = query_emb.to(torch.float16)  #[L, D]

        emb_files = sorted(glob.glob(os.path.join("D:/data_process/test", "emb_*.pt")), key=extract_number)
        # total_sim = defaultdict(list)
        total_sim = {query_id: torch.empty(post_cnt, device=DEVICE) for query_id in query_embs}
        sim_list = []   #to store the similarity between the B formulas in the current query and all pool docs
        doc_base_idx = 0
        for path in tqdm(emb_files, desc="loading embeddings"):
            embs_pooling = torch.load(path, map_location='cpu') #list of tensors, each tensor is [L', D], which is the tensor of one pool doc
            count = 1
            for i, doc_emb in enumerate(embs_pooling):
                doc_emb = doc_emb.to(DEVICE)
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
        print(doc_base_idx)
        torch.save(total_sim, cfg.DATA.SIM)

    # emb_files = sorted(glob.glob(os.path.join(cfg.DATA.EMBS_FILE, "emb_*.pt")), key=extract_number)
    # from typing import List, Tuple
    # def pad_embeddings(emb_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Pad a list of [L, D] tensors to shape [B, max_L, D], with mask.
    #     """
    #     B = len(emb_list)
    #     D = emb_list[0].shape[-1]
    #     lengths = [emb.shape[0] for emb in emb_list]
    #     max_L = max(lengths)

    #     padded = torch.zeros((B, max_L, D), dtype=emb_list[0].dtype, device=emb_list[0].device)
    #     mask = torch.zeros((B, max_L), dtype=torch.bool, device=emb_list[0].device)

    #     for i, emb in enumerate(emb_list):
    #         L = emb.shape[0]
    #         padded[i, :L] = emb
    #         mask[i, :L] = 1

    #     return padded, mask
    # total_sim = defaultdict(list)
    # for query_id, query_emb in tqdm(query_embs.items(), desc="Processing queries"):
    #     query_emb = query_emb.to(DEVICE)              # [Lq, D]
    #     for path in emb_files:
    #         doc_list = torch.load(path, map_location=DEVICE)  # list of [Ld, D]
    #         # doc_list = [d.to(DEVICE) for d in doc_list]
    #         doc_stack, doc_mask = pad_embeddings(doc_list)    # [N, Ld, D], [N, Ld]
    #         doc_stack = doc_stack.to(DEVICE)
    #         doc_mask = doc_mask.to(DEVICE)  
    #         # einsum: [N, Ld, D] · [Lq, D]ᵗ → [N, Ld, Lq]
    #         sim = torch.matmul(doc_stack, query_emb.T)

    #         # mask out padded doc tokens
    #         sim = sim.masked_fill(~doc_mask.unsqueeze(dim=2), -1e4)

    #         # MaxSim over doc tokens
    #         sim = sim.max(dim=1).values         # [N, Lq]

    #         # mean over query tokens
    #         # sim = sim.mean(dim=1)     # [N]
    #         total_sim[query_id].extend(sim.tolist())  # [N]


    




    # total_topk = {}   #old rank method
    # for query_id, sim_list in total_sim.items():
    #     sim_tensor = torch.tensor(sim_list)
    #     topk_sim, topk_idx = torch.topk(sim_tensor, k=cfg.INF.TOPK)
    #     real_topk_idx = [pool_id[i] for i in topk_idx.tolist()]
    #     total_topk[query_id] = {"sim": topk_sim,
    #                             "post_idx": real_topk_idx
    #                             }
        
    total_sim = torch.load(cfg.DATA.SIM, map_location=DEVICE)
    total_topk = {}
    for query_id, sim_tensor in total_sim.items():  # sim_tensor 是 GPU tensor [1300000]
        topk_sim, topk_idx = torch.topk(sim_tensor, k=cfg.INF.TOPK)

        # 如果 pool_id 是 list，可以提前转成 tensor
        pool_id_tensor = torch.tensor(pool_id, device=topk_idx.device)

        real_topk_idx = pool_id_tensor[topk_idx]  # ✅ GPU 上索引，无需 .tolist()

        total_topk[query_id] = {
            "sim": topk_sim,
            "post_idx": real_topk_idx
        }

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
        
