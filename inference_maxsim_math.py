from transformers import BertModel, BertConfig, BertTokenizer
from config import get_config, DEVICE, SEED
import logger
from torch import Tensor
from dataset import TOPIC, POOLING
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from transformer import Transformer
from collections import defaultdict
from emb import embedding, embedding_maxsim_math
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd
import re
import glob
import os
from tqdm import tqdm

def main() -> None:
    cfg = get_config(args=None)
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.CKPT.BERT.TOK
    )
    print(0)
    config = BertConfig.from_json_file(cfg.CKPT.BERT.CFG)
    model = BertModel(config, add_pooling_layer=False)
    # model = Transformer(    #TODO: confirm arguments of model
    #     vocab_size=len(tokenizer.vocabs),
    #     dim=cfg.MODEL.TX.DIM,
    #     n_layers=cfg.MODEL.TX.N_LAYERS,
    #     n_heads=cfg.MODEL.TX.N_HEADS,
    #     n_kv_heads=cfg.MODEL.TX.N_KV_HEADS,
    #     base=cfg.MODEL.TX.BASE,
    #     max_seq_len=cfg.MODEL.TX.MAX_SEQ_LEN,
    #     multiple_of=cfg.MODEL.TX.MULTIPLE_OF,
    #     ffn_dim_multiplier=cfg.MODEL.TX.FFN_DIM_MULTIPLIER,
    #     norm_eps=cfg.MODEL.TX.NORM_EPS,
    # )

    pool_id = embedding_maxsim_math(    #uncomment this for embs generation
        model=model,
        device=DEVICE,
        file_path = cfg.DATA.DICT_FILE,
        ckpt_filepath=cfg.CKPT.TX.LAST,
        tokenizer=tokenizer,
        emb_filepate=cfg.DATA.EMBS_FILE,
    )   #emb_pooling: num of formulas in pool x emb_size

    logger.log_info("Generate formula embeddings...")
    model.to(device=DEVICE)
    linear = nn.Linear(config.hidden_size, 128, bias=False).to(device=DEVICE)
    model.eval()
    state_dict = torch.load(cfg.CKPT.BERT.PT, map_location="cuda")
    model.load_state_dict(state_dict=state_dict["model_state_dict"], strict=False)
    logger.log_info(f"Loaded model '{cfg.CKPT.BERT.PT}'")
    ################# load query docs#######################
    exprs = []
    file = open(file=cfg.DATA.TOPIC_FILE, mode='r', encoding='utf-8')
    for line in file:
        line = line.strip().split(sep='\t')
        exprs.append(line)
    file.close()
    query_docs = defaultdict(list)  #{"topic_id": [formula1, formula2...]}
    for line in exprs:
        query_id = line[1]
        query_docs[query_id].append(line[0])

    def extract_number(filename):
        match = re.search(r'emb_(\d+)\.pt', filename)
        return int(match.group(1)) if match else float('inf')
    
    total_sim = defaultdict(list)
    ########################## load and compuate query embs #############################
    query_embs = {}
    with torch.no_grad():
        for query_id, query_list in tqdm(query_docs.items(), desc="loading topics"):
            exprs = [expr for expr in query_list]
            # src = [tokenizer.encode(expr=expr) for expr in query_formula_list]
            # src = pad_sequence(
            #         sequences=src,
            #         batch_first=True,
            #         padding_value=tokenizer.word2idx["PAD"],
            #     ).to(device=DEVICE)
            # src_mask = torch.eq(input=src, other=tokenizer.word2idx["PAD"]) \
            #     .unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.bool)
            # src_mask = src_mask.to(device=DEVICE)
            # query_emb = model(tokens=src, mask=src_mask, cache_pos=None).to(torch.float16)    #[B, L, D] embedding of formulas under the current query doc
            tokens = tokenizer(
                    text=exprs,
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=cfg.MODEL.BERT.MAX_SEQ_LEN,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
            input_ids = tokens["input_ids"].to(device=DEVICE)
            attn_mask = tokens["attention_mask"].to(device=DEVICE, dtype=torch.bool)
            query_emb = model(input_ids=input_ids, attention_mask=attn_mask)    #[B, L, D] embedding of formulas under the current query doc
            query_emb = query_emb.last_hidden_state
            query_emb = linear(query_emb)  
            # src_mask = src_mask.squeeze(dim=(-3, -2))
            # n_pad = src_mask.int().sum(dim=-1)
            # eoe_ids = src_mask.size(dim=-1) - n_pad - 1
            # batch_ids = torch.arange(
            #     start=0, end=src_mask.size(dim=0), dtype=torch.int64, device=src_mask.device
            # )
            # src_mask[batch_ids, eoe_ids] = True
            # src_mask[:, 0] = True
            # query_emb = query_emb * (~src_mask).unsqueeze(dim=-1).float()
            # query_emb = F.normalize(input=query_emb, p=2.0, dim=-1, eps=1e-12).detach().cpu()
            # query_embs[query_id] = query_emb
            pad_mask = attn_mask == 0
            n_pad = pad_mask.int().sum(dim=-1)
            eoe_ids = attn_mask.size(dim=-1) - n_pad - 1
            batch_ids = torch.arange(
                start=0, end=attn_mask.size(dim=0), dtype=torch.int64, device=attn_mask.device
            )
            attn_mask[batch_ids, eoe_ids] = True
            attn_mask[:, 0] = True
            query_emb = query_emb * (~attn_mask).unsqueeze(dim=-1).float()
            # query_emb = query_emb.mean(dim=0)
            query_emb = F.normalize(input=query_emb, p=2.0, dim=-1, eps=1e-12)
            query_embs[query_id] = query_emb.to(torch.float16)

    emb_files = sorted(glob.glob(os.path.join(cfg.DATA.EMBS_FILE, "emb_*.pt")), key=extract_number)
    sim_list = []   #to store the similarity between the B formulas in the current query and all pool docs
    for path in tqdm(emb_files, desc="loading embeddings"):
        embs_pooling = torch.load(path) #list of tensors, each tensor is [L', D], which is the tensor of one pool doc
        for doc_emb in embs_pooling:
            doc_emb = doc_emb.to(DEVICE)
            for query_id, query_emb in query_embs.items():
                sim = query_emb @ doc_emb.T #[B, L, L'], L is seqlen of query, L' is seqlen of document
                sim = torch.max(input=sim, dim=-1).values 
                sim = sim.sum(dim=-1) #[B]
                total_sim[query_id].append(sim.sum(dim=0))  #when loop is done, total_sim[query_id] is a list of [num of pool doc]



    total_topk = {}
    for query_id, sim_tensor in total_sim.items():
        sim_tensor = torch.tensor(sim_list, device=DEVICE)
        topk_sim, topk_idx = torch.topk(sim_tensor, k=cfg.INF.TOPK)
        real_topk_idx = [pool_id[i] for i in topk_idx.tolist()]
        total_topk[query_id] = {"sim": topk_sim,
                                "post_idx": real_topk_idx
                                }
    print(2)


    rows = []
    for query_id, data in total_topk.items():
        sims = data["sim"]
        doc_ids = data["post_idx"]
        for rank, (doc_id, sim) in enumerate(zip(doc_ids, sims), start=1):
            rows.append({
                "Query_Id": query_id,
                "Post_Id": doc_id,
                "Rank": rank,
                "Score": sim,
                "Run_Number": "Run_0" 
            })
    df = pd.DataFrame(rows)
    df.to_csv(cfg.DATA.PRED, index=False, sep='\t')

if __name__ == '__main__':
    main()     
        
