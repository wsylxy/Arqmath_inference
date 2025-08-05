from config import get_config, DEVICE, SEED
import logger
from torch import Tensor
from dataset import TOPIC, POOLING
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from transformer import Transformer
from emb import embedding, pool_embedding
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
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
    # print(tokenizer.word2idx)
    #load data
    topic = TOPIC(file_path=cfg.DATA.TOPIC_FILE, tokenizer=tokenizer)
    # print(topic[0])
    topic_loader = DataLoader(
        dataset=topic,
        batch_size=cfg.LOADER.INF.BATCH_SIZE,
        shuffle=cfg.LOADER.INF.SHUFFLE,
        num_workers=cfg.LOADER.INF.NUM_WORKERS,
        collate_fn=topic.collate_fn,
        pin_memory=cfg.LOADER.INF.PIN_MEMORY,
        )
    pooling = POOLING(file_path=cfg.DATA.DICT_FILE, tokenizer=tokenizer)    #uncomment this for embs generation
    print(0)
    pooling_loader = DataLoader(    #uncomment this for embs generation
        dataset=pooling,
        batch_size=cfg.LOADER.INF.BATCH_SIZE,
        shuffle=cfg.LOADER.INF.SHUFFLE,
        num_workers=cfg.LOADER.INF.NUM_WORKERS,
        collate_fn=pooling.collate_fn,
        pin_memory=cfg.LOADER.INF.PIN_MEMORY,
        )
    id_pooling = []
    file = pd.read_csv(cfg.DATA.DICT_FILE, encoding='utf-8')
    for i, row in file.iterrows():
        id_pooling.append(row["post_id"])

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
    
    embs_topic, id_topic = embedding(
        model=model,
        device=DEVICE,
        ckpt_filepath=cfg.CKPT.TX.LAST,
        data_loader=topic_loader,
        mode=cfg.INF.MODE,
    )
    print(2)
    topics = {} #{id: [formula_emb1, formula_emb2...]}
    for i, id in enumerate(id_topic):
        if id not in topics:
            topics[id] = []
            topics[id].append(embs_topic[i])
        else:
            topics[id].append(embs_topic[i])
    
    # print(topics)
    # print(3)
    id_pooling = pool_embedding(    #uncomment this for embs generation
        model=model,
        device=DEVICE,
        ckpt_filepath=cfg.CKPT.TX.LAST,
        data_loader=pooling_loader,
        emb_filepate=cfg.DATA.EMBS_FILE,
        mode=cfg.INF.MODE,
    )   #emb_pooling: num of formulas in pool x emb_size

    def extract_number(filename):
        match = re.search(r'emb_(\d+)\.pt', filename)
        return int(match.group(1)) if match else float('inf')

    
    top_k = {}  #{topic_id1: [('doc_id1', sim1), ('doc_id2', sim2)], ...}
    for topic_id, topic_emb_list in tqdm(topics.items(), desc="loading topics"):
        topic_topk = {} #top k pool formula id and their similarity of the current topic
        topic_embs = torch.stack(topic_emb_list)
        topic_embs = F.normalize(topic_embs, dim=1)
        files = sorted(glob.glob(os.path.join(cfg.DATA.EMBS_FILE, "emb_*.pt")), key=extract_number)
        sims = []
        for path in files:
            embs_pooling = torch.load(path)
            embs_pooling = F.normalize(embs_pooling, dim=1)
            sims.append(topic_embs @ embs_pooling.T)    
        sim = torch.cat(sims, dim=1)    #[num of formulas in the current topic, num of pool formulas]

    ####################### Choice: sum similarity for all formulas under the current query #######################
        sim = sim.sum(dim=0, keepdim=True).squeeze(0)  #[num of pool formulas]
        doc_sim_sum = defaultdict(float)
        for sim_value, doc_id in zip(sim.tolist(), id_pooling):
            doc_sim_sum[doc_id] += sim_value
        sorted_docs = sorted(doc_sim_sum.items(), key=lambda x: x[1], reverse=True)
        topk_docs = sorted_docs[:cfg.INF.TOPK]  #[('doc_id1', sim1), ('doc_id2', sim2)]
        top_k[topic_id] = topk_docs

    rows = []
    for query_id, data in top_k.items():
        for rank, (post_id, sim) in enumerate(data, start=1):
            rows.append({
                "Query_Id": query_id,
                "Post_Id": post_id,
                "Rank": rank,
                "Score": sim,
                "Run_Number": "Run_0" 
            })

    #################### Choice: topk for each query formula ####################
    #     topk_sim, topk_idx = torch.topk(sim, k=cfg.INF.TOPK, dim=1) #topk_sim: num_formula in under this topic x k
    #     for i in range(topk_sim.shape[0]):
    #         for j in range(topk_sim.shape[1]):
    #             pool_formula_id = id_pooling[topk_idx[i][j]]
    #             if pool_formula_id not in topic_topk:
    #                 topic_topk[pool_formula_id] = topk_sim[i][j].item()
    #             elif pool_formula_id in topic_topk:
    #                 topic_topk[pool_formula_id] += topk_sim[i][j].item()
    #     topic_topk = dict(sorted(topic_topk.items(), key=lambda item: item[1], reverse=True)[:cfg.INF.TOPK])
    #     top_k[topic_id] = topic_topk
    # rows = []
    # for query_id, post_id in top_k.items():
    #     for rank, (post_id, score) in enumerate(post_id.items(), start=1):
    #         rows.append({
    #             "Query_Id": query_id,
    #             "Post_Id": post_id,
    #             "Rank": rank,
    #             "Score": score,
    #             "Run_Number": "Run_0" 
    #         })
    ######################################################################

    df = pd.DataFrame(rows)
    df.to_csv(cfg.DATA.PRED, index=False, sep='\t')

if __name__ == '__main__':
    main()     
        
