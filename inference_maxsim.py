from config import get_config, DEVICE, SEED
import logger
from torch import Tensor
from dataset import TOPIC, POOLING
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from transformer import Transformer
from collections import defaultdict
from emb import embedding, pooling_maxsim_post
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import torch.nn.functional as F
import torch
import pandas as pd
import re
import glob
import os
from tqdm import tqdm

def main() -> None:
    cfg = get_config(args=None)
    tokenizer = Tokenizer(file_path=cfg.DATA.VOCAB_FILE)
    print(0)
    model = Transformer(    #TODO: confirm arguments of model
        vocab_size=len(tokenizer.vocabs),
        dim=cfg.MODEL.TX.DIM,
        n_layers=cfg.MODEL.TX.N_LAYERS,
        n_heads=cfg.MODEL.TX.N_HEADS,
        n_kv_heads=cfg.MODEL.TX.N_KV_HEADS,
        base=cfg.MODEL.TX.BASE,
        max_seq_len=cfg.MODEL.TX.MAX_SEQ_LEN,
        multiple_of=cfg.MODEL.TX.MULTIPLE_OF,
        ffn_dim_multiplier=cfg.MODEL.TX.FFN_DIM_MULTIPLIER,
        norm_eps=cfg.MODEL.TX.NORM_EPS,
    )

    pool_id = pooling_maxsim_post(    #uncomment this for embs generation
        model=model,
        device=DEVICE,
        file_path = cfg.DATA.DICT_FILE,
        ckpt_filepath=cfg.CKPT.TX.LAST,
        tokenizer=tokenizer,
        emb_filepate=cfg.DATA.EMBS_FILE,
    )   #emb_pooling: num of formulas in pool x emb_size

    logger.log_info("Generate formula embeddings...")
    model.to(device=DEVICE)
    model.eval()
    ckpt = torch.load(f=cfg.CKPT.TX.LAST, map_location=DEVICE)
    model.load_state_dict(state_dict=ckpt["model_state"])
    logger.log_info(f"Loaded model '{cfg.CKPT.TX.LAST}'")
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
    for query_id, query_formula_list in tqdm(query_docs.items(), desc="loading topics"):
        src = [tokenizer.encode(expr=expr) for expr in query_formula_list]
        src = pad_sequence(
                sequences=src,
                batch_first=True,
                padding_value=tokenizer.word2idx["PAD"],
            ).to(device=DEVICE)
        src_mask = torch.eq(input=src, other=tokenizer.word2idx["PAD"]) \
            .unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.bool)
        src_mask = src_mask.to(device=DEVICE)
        query_emb = model(tokens=src, mask=src_mask, cache_pos=None).to(torch.float16)    #[B, L, D] embedding of formulas under the current query doc
        src_mask = src_mask.squeeze(dim=(-3, -2))
        n_pad = src_mask.int().sum(dim=-1)
        eoe_ids = src_mask.size(dim=-1) - n_pad - 1
        batch_ids = torch.arange(
            start=0, end=src_mask.size(dim=0), dtype=torch.int64, device=src_mask.device
        )
        src_mask[batch_ids, eoe_ids] = True
        src_mask[:, 0] = True
        query_emb = query_emb * (~src_mask).unsqueeze(dim=-1).float()
        query_emb = F.normalize(input=query_emb, p=2.0, dim=-1, eps=1e-12).detach().cpu()
        ####################### load pool embs#########################
        emb_files = sorted(glob.glob(os.path.join(cfg.DATA.EMBS_FILE, "emb_*.pt")), key=extract_number)
        sim_list = []   #to store the similarity between the B formulas in the current query and all pool docs
        for path in emb_files:
            embs_pooling = torch.load(path) #list of tensors with shape [B], list of tensors, each tensor is [L, D], which is the tensor of one pool doc
            for doc_emb in embs_pooling:
                sim = query_emb @ doc_emb.T
                sim = torch.max(input=sim, dim=-1).values 
                # seqlen = sim.shape[1]
                sim = sim.sum(dim=-1, keepdim=True) #[B]
                sim_list.append(sim.unsqueeze(1))
        sim_list = torch.cat(sim_list, dim=1)   #[B, num of pool doc]
        sim_list = sim_list.sum(dim=0)  #[num of pool doc], similarity between the current query and all pool docs
        total_sim[query_id].extend(sim_list)  #.extend here will break the tenor sim_list and add every elements to the list in order

    total_topk = {}
    for query_id, sim_tensor in total_sim.items():
        topk_sim, topk_idx = torch.topk(sim_tensor, k=cfg.INF.TOPK, dim=0)
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
        
