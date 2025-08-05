from torch import Tensor
from transformers import BertModel, BertConfig
from config import get_config, DEVICE, SEED

import logger
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from logger import timestamp
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizer import Tokenizer
from collections import defaultdict
from tqdm import tqdm

cfg = get_config(args=None)

def embedding(
        model: nn.Module,
        device: torch.device,
        ckpt_filepath: str,
        data_loader: DataLoader,
        mode: str,
) -> tuple[Tensor, list]:
    logger.log_info("Generate formula embeddings...")
    model.to(device=device)
    model.eval()
    ckpt = torch.load(f=ckpt_filepath, map_location=device)
    model.load_state_dict(state_dict=ckpt["model_state"])
    print("loss:", ckpt["loss"])
    logger.log_info(f"Loaded model '{ckpt_filepath}'")
    # print(0.5)
    embs = []
    ids = []
    loader_tqdm = tqdm(iterable=data_loader, position=1, leave=False)
    loader_tqdm.set_description(desc=f"[{timestamp()}] [Batch 0]", refresh=True)
    for i, batch in enumerate(loader_tqdm):
        src = batch["src"].to(device=device)
        src_mask = batch["src_mask"].to(device=device)
        id = batch["id"]
        ids.extend(id)
        emb = model(tokens=src, mask=src_mask, cache_pos=None).to(torch.float16)
        # print(0.7)
        src_mask = src_mask.squeeze(dim=(-3, -2))
        # src_mask[:, 0] = 0
        # last_1 = src_mask.sum(dim=1)
        # src_mask[torch.arange(src_mask.size(dim=0)), last_1] = 0

        n_pad = src_mask.int().sum(dim=-1)
        eoe_ids = src_mask.size(dim=-1) - n_pad - 1
        batch_ids = torch.arange(
            start=0, end=src_mask.size(dim=0), dtype=torch.int64, device=src_mask.device
        )
        src_mask[batch_ids, eoe_ids] = True
        src_mask[:, 0] = True

        if mode == "mean":
            emb[src_mask] = 0.0
            # emb = emb.masked_fill(src_mask.unsqueeze(-1), 0.0)
            emb = emb.mean(dim=-2, keepdim=False)
            # emb[src_mask==0] = 0
            # emb = emb.sum(dim=-2)
            # last_1 = last_1.unsqueeze(dim=1)
            # emb /= last_1
        elif mode == "max":
            emb[src_mask==0] = float("-inf")
            emb, _ = emb.max(dim=1, keepdim=False)
        elif mode == "start":
            emb = emb[:, 0, :]
        else:
            logger.log_error("Invalid mode for embedding!")
        
        embs.append(emb.detach().cpu())
    
    embs = torch.cat(tensors=embs, dim=0)
    logger.log_info("Finish generating expression embeddings")
    return embs, ids

def pool_embedding(
        model: nn.Module,
        device: torch.device,
        ckpt_filepath: str,
        data_loader: DataLoader,
        emb_filepate: str,
        mode: str,
) -> tuple[Tensor, list]:
    logger.log_info("Generate formula embeddings...")
    model.to(device=device)
    model.eval()
    ckpt = torch.load(f=ckpt_filepath, map_location=device)
    model.load_state_dict(state_dict=ckpt["model_state"])
    logger.log_info(f"Loaded model '{ckpt_filepath}'")
    # print(0.5)
    embs = []
    ids = []
    loader_tqdm = tqdm(iterable=data_loader, position=1, leave=False)
    loader_tqdm.set_description(desc=f"[{timestamp()}] [Batch 0]", refresh=True)
    for i, batch in enumerate(loader_tqdm):
        src = batch["src"].to(device=device)
        src_mask = batch["src_mask"].to(device=device)
        id = batch["id"]
        ids.extend(id)
        emb = model(tokens=src, mask=src_mask, cache_pos=None).to(torch.float16)
        # print(0.7)
        src_mask = src_mask.squeeze(dim=(-3, -2))
        # src_mask[:, 0] = 0
        # last_1 = src_mask.sum(dim=1)
        # src_mask[torch.arange(src_mask.size(dim=0)), last_1] = 0

        n_pad = src_mask.int().sum(dim=-1)
        eoe_ids = src_mask.size(dim=-1) - n_pad - 1
        batch_ids = torch.arange(
            start=0, end=src_mask.size(dim=0), dtype=torch.int64, device=src_mask.device
        )
        src_mask[batch_ids, eoe_ids] = True
        src_mask[:, 0] = True

        if mode == "mean":
            emb[src_mask] = 0.0
            emb = emb.mean(dim=-2, keepdim=False)
            # emb[src_mask==0] = 0
            # emb = emb.sum(dim=-2)
            # last_1 = last_1.unsqueeze(dim=1)
            # emb /= last_1
        elif mode == "max":
            emb[src_mask==0] = float("-inf")
            emb, _ = emb.max(dim=1, keepdim=False)
        elif mode == "start":
            emb = emb[:, 0, :]
        else:
            logger.log_error("Invalid mode for embedding!")
        
        embs.append(emb.detach().cpu())
        if i%20000==0 and i!=0:
            file_idx = i/20000
            emb_file = f"emb_{str(file_idx)}.pt"
            embs = torch.cat(tensors=embs, dim=0)
            torch.save(embs, f'{emb_filepate}{emb_file}')
            embs = []

    if len(embs) != 0:  #write the last emb file
        file_idx = file_idx+1
        emb_file = f"emb_{str(file_idx)}.pt"
        embs = torch.cat(tensors=embs, dim=0)
        torch.save(embs, f'{emb_filepate}{emb_file}')
    
    logger.log_info("Finish generating expression embeddings")
    return ids

def embedding_maxsim_math(
        model: nn.Module,
        device: torch.device,
        file_path: str,
        ckpt_filepath: str,
        tokenizer: Tokenizer,
        emb_filepate: str,
):
    logger.log_info("Generate formula embeddings...")
    
    model.to(device=device)
    model.eval()
    ckpt = torch.load(f=ckpt_filepath, map_location=device)
    model.load_state_dict(state_dict=ckpt["model_state"])
    logger.log_info(f"Loaded model '{ckpt_filepath}'")

    file = pd.read_csv(file_path, encoding='utf-8')
    docs = defaultdict(list)    #{"doc_id": [formula1, formula2...]}
    for id, row in file.iterrows():
        doc_id = row["post_id"]
        docs[doc_id].append(row["formula"])
    print("number of post is", len(docs))
    embs = []

    ids = []
    for i, (key, value) in enumerate(docs.items()):
        ids.append(key)
        src = [tokenizer.encode(expr=expr) for expr in value]
        src = pad_sequence(
                sequences=src,
                batch_first=True,
                padding_value=tokenizer.word2idx["PAD"],
            ).to(device=device)
        src_mask = torch.eq(input=src, other=tokenizer.word2idx["PAD"]) \
            .unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.bool)
        src_mask = src_mask.to(device=device)
        emb = model(tokens=src, mask=src_mask, cache_pos=None).to(torch.float16)

        src_mask = src_mask.squeeze(dim=(-3, -2))
        n_pad = src_mask.int().sum(dim=-1)
        eoe_ids = src_mask.size(dim=-1) - n_pad - 1
        batch_ids = torch.arange(
            start=0, end=src_mask.size(dim=0), dtype=torch.int64, device=src_mask.device
        )
        src_mask[batch_ids, eoe_ids] = True
        src_mask[:, 0] = True

        emb = emb * (~src_mask).unsqueeze(dim=-1).float()
        emb = emb.mean(dim=0)
        emb = F.normalize(input=emb, p=2.0, dim=-1, eps=1e-12)
        embs.append(emb.detach().cpu())
        if i%5000==0 and i!=0:
            file_idx = i/5000
            emb_file = f"emb_{str(file_idx)}.pt"
            torch.save(embs, f'{emb_filepate}{emb_file}')
            embs = []
    return ids

# def embedding_maxsim_post_old(
#         model,
#         config,
#         device: torch.device,
#         file_path: str,
#         ckpt_filepath: str,
#         tokenizer,
#         emb_filepate: str,
# ):
#     logger.log_info("Generate formula embeddings...")
#     model.to(device=device)
#     linear = nn.Linear(config.hidden_size, 128, bias=False).to(device=device)
#     model.eval()
#     state_dict = torch.load(ckpt_filepath, map_location="cuda")
#     model.load_state_dict(state_dict=state_dict["model_state_dict"], strict=False)
#     logger.log_info(f"Loaded model '{ckpt_filepath}'")

#     file = pd.read_csv(file_path, encoding='utf-8')
#     docs = {}   #{"doc_id": [formula1, formula2...]}
#     for id, row in file.iterrows():
#         doc_id = row["post_id"]
#         docs[doc_id] = row["doc"]
#     # print("number of post is", len(docs))
#     embs = []
#     length = []
#     ids = []
#     for i, (key, post) in enumerate(tqdm(docs.items())):
#         # print("this is the post", post)
#         # print("it's a type of", type(post))
#         if type(post) != str or len(post) < 10:
#             continue
#         ids.append(key)
#         tokens = tokenizer(
#             text=post,
#             add_special_tokens=True,
#             padding=True,
#             truncation=True,
#             max_length=cfg.MODEL.BERT.MAX_SEQ_LEN,
#             return_tensors="pt",
#             return_attention_mask=True,
#         )
#         # print(tokenizer.tokenize(post))
        
#         # print(tokens)
#         input_ids = tokens["input_ids"].to(device=DEVICE)
#         # print(input_ids)
#         # print(input_ids.shape)
#         length.append(input_ids.shape[1])
#         attn_mask = tokens["attention_mask"].to(device=DEVICE)
#         # print(input_ids.shape)
#         # print(attn_mask.shape)
#         doc_emb = model(input_ids=input_ids, attention_mask=attn_mask)
#         doc_emb = doc_emb.last_hidden_state
#         doc_emb = linear(doc_emb)
#         # print(query_emb.shape)
#         pad_mask = attn_mask == 0
#         n_pad = pad_mask.int().sum(dim=-1)
#         eoe_ids = attn_mask.size(dim=-1) - n_pad - 1
#         # print(f"attn_mask.shape: {attn_mask.shape}, n_pad: {n_pad}, eoe_ids: {eoe_ids}")
#         batch_ids = torch.arange(
#             start=0, end=attn_mask.size(dim=0), dtype=torch.int64, device=attn_mask.device
#         )
#         attn_mask[batch_ids, eoe_ids] = False
#         attn_mask[:, 0] = False
#         doc_emb = doc_emb * (attn_mask).unsqueeze(dim=-1).float()
#         # query_emb[attn_mask] = 0.0
#         doc_emb = doc_emb.squeeze(dim=0)
#         # print(query_emb)
#         # print(query_emb.shape)
#         # print(torch.isnan(query_emb).any(), torch.isinf(query_emb).any())
#         # print(query_emb.min(), query_emb.max())
#         # print(query_emb)
#         doc_emb = F.normalize(input=doc_emb, p=2.0, dim=-1, eps=1e-12).to(torch.float16)
#         # print(query_emb.dtype)
#         embs.append(doc_emb.detach().cpu())
#         if i%5000==0 and i!=0:
#             # print("mean_len", sum(length)/len(length))
#             file_idx = i/5000
#             emb_file = f"emb_{str(file_idx)}.pt"
#             torch.save(embs, f'{emb_filepate}{emb_file}')
#             embs = []
#     file_idx = i%5000 + 1
#     emb_file = f"emb_{str(file_idx)}.pt"
#     torch.save(embs, f'{emb_filepate}{emb_file}')
#     embs = []
#     return ids, len(ids)

def embedding_maxsim_post_read_file(
        file_path: str,
        emb_filepate: str,
        encoder,
):
    logger.log_info("Generate formula embeddings...")
    file = pd.read_csv(file_path, encoding='utf-8')
    docs = {}   #{"doc_id": [formula1, formula2...]}
    for id, row in file.iterrows():
        doc_id = row["post_id"]
        docs[doc_id] = row["doc"]
    # print("number of post is", len(docs))
    embs = []
    ids = []
    for i, (key, post) in enumerate(tqdm(docs.items())):
        # print("this is the post", post)
        # print("it's a type of", type(post))
        # if type(post) != str or len(post) < 10:
        #     continue
        ids.append(key)
        doc_emb, _ = encoder([post])
        # print(doc_emb.shape)
        embs.append(doc_emb.detach().cpu())
        # embs.append(doc_emb.to(torch.float16).detach().cpu())
        if i%5000==0 and i!=0:
            # print("mean_len", sum(length)/len(length))
            file_idx = i/5000
            emb_file = f"emb_{str(file_idx)}.pt"
            torch.save(embs, f'{emb_filepate}{emb_file}')
            embs = []
    file_idx = i//5000 + 1
    emb_file = f"emb_{str(file_idx)}.pt"
    torch.save(embs, f'{emb_filepate}{emb_file}')
    embs = []
    return ids, len(ids)

def embedding_maxsim_post(  #完全用原作者的corpus reader 和 doc process
        xml_file_path: str,
        emb_filepate: str,
        encoder,
):
    from maxsim.corpus_reader import corpus_reader__arqmath3_rawxml, corpus_length__arqmath3_rawxml
    import sys
    logger.log_info("Generate formula embeddings...")
    corpus_reader_begin = 0
    corpus_reader_end = 0
    corpus_max_reads = corpus_reader_end - corpus_reader_begin
    batch_sz = 6
    save_every = 12  # 每12个batch存一次
    save_idx = 0 
    n = corpus_length__arqmath3_rawxml(xml_file_path, corpus_max_reads)
    if n is None: n = 0
    print('corpus length:', n)
    progress = tqdm(corpus_reader__arqmath3_rawxml(xml_file_path), total=n)
    batch = []
    all_embs, all_lengths, all_doc_ids = [], [], []
    batch_cnt = 0
    doc_cnt = 0
    skip_cnt = 0
    for row_idx, doc in enumerate(progress):
        # print(skip_cnt)
        # doc is of ((docid, *doc_props), doc_content)
        if doc[1] is None:
            skip_cnt+=1
            continue # Task1 Question is skipped
        if row_idx < corpus_reader_begin:
            skip_cnt+=1
            continue
        elif corpus_reader_end > 0 and row_idx >= corpus_reader_end:
            break
        batch.append(doc)   # batch is of [((docid, *doc_props), doc_content), ...]
        if len(batch) == batch_sz:
            doc_ids = [doc[0][0] for doc in batch]
            doc_cnt += len(doc_ids)
            passages = [psg for docid, psg in batch]
            embs, lengths = encoder(passages)
            sys.exit()
            # lengths = lengths.tolist()
            embs = embs.to(torch.float16)
            for b in range(len(doc_ids)):
                valid_len = lengths[b]
                all_embs.append(embs[b, :valid_len])         # embs: Tensor[B, L_i, D]
                all_lengths.append(valid_len)   # lengths: a list
                all_doc_ids.append(doc_ids[b])   # list of B doc ids
            # doc_cnt += len(doc_ids)
            batch = []
            batch_cnt += 1
            if batch_cnt == 1:
                sys.exit("Program stopped here.")
            if batch_cnt%save_every==0 and batch_cnt!=0:
                emb_tensor = torch.cat(all_embs, dim=0)
                emb_save = {
                    "doc_ids": all_doc_ids,
                    "embs": emb_tensor,
                    "lengths": all_lengths
                }
                torch.save(emb_save, f"{emb_filepate}emb_{save_idx}.pt")
                print(f"Saved final: {emb_filepate}emb_{save_idx}.pt")
                save_idx += 1
                all_embs, all_lengths, all_doc_ids = [], [], []

    if batch or all_embs:
        if batch:
            doc_ids = [doc[0][0] for doc in batch]
            doc_cnt += len(doc_ids)
            passages = [doc[1] for doc in batch]
            embs, lengths = encoder(passages)
            embs = embs.to(torch.float16)
            for b in range(len(doc_ids)):
                valid_len = lengths[b]
                all_embs.append(embs[b, :valid_len])
                all_lengths.append(valid_len)
                all_doc_ids.append(doc_ids[b])
            # doc_cnt += len(doc_ids)
        emb_tensor = torch.cat(all_embs, dim=0)
        emb_save = {
            "doc_ids": all_doc_ids,
            "embs": emb_tensor,
            "lengths": all_lengths
        }
        torch.save(emb_save, f"{emb_filepate}emb_{save_idx}.pt")
        print(f"Saved final: {emb_filepate}emb_{save_idx}.pt")
    return doc_cnt

def embedding_maxsim_formula(  #完全用原作者的corpus reader 和 doc process的纯数学版
        xml_file_path: str,
        emb_filepate: str,
        encoder,
):
    import sys
    logger.log_info("Generate formula embeddings...")
    save_every = 10000  # 每10000个doc存一次
    save_idx = 0 
    batch = []
    all_embs, all_lengths, all_doc_ids = [], [], []
    batch_cnt = 0
    doc_cnt = 0
    docs = defaultdict(list)
    file = pd.read_csv(cfg.DATA.DICT_FILE, encoding='utf-8')
    # p = ["$T$ $equal$ $\{$ $x$ $in$ $X$ $colon$ $x$ $in$ $f$ $($ $x$ $)$ $\}$", 
    #      "$0$ $comma$ $q$ $subscript$ $1$ $comma$ $minus$ $q$ $subscript$ $1$ $comma$ $q$ $subscript$ $2$ $comma$ $minus$ $q$ $subscript$ $2$ $comma$ $q$ $subscript$ $3$ $comma$ $minus$ $q$ $subscript$ $3$ $comma$"]
    # embs, lengths = encoder(p)
    # sys.exit()
    for id, row in file.iterrows():
        doc_id = row["doc_id"]
        docs[doc_id].append(row["formula"])
        # print(skip_cnt)
        # doc is of ((docid, *doc_props), doc_content)
    for doc_id, doc in tqdm(docs.items()):
        batch = [formula for formula in doc]   # batch is of [((docid, *doc_props), doc_content), ...]
        doc_cnt += 1
        embs, lengths = encoder(batch)
        # lengths = lengths.tolist()
        # max_len = max(lengths)
        embs = embs.to(torch.float16)
        # print(embs)
        # print(embs.shape)
        # print(lengths)
        assert len(lengths) == 1
        all_embs.append(embs[0])         # embs: Tensor[1, L_i, D]
        all_lengths.append(lengths[0])   # length: int
        all_doc_ids.append(doc_id)
        # doc_cnt += len(doc_ids)
        batch = []
        batch_cnt += 1

        if batch_cnt%save_every==0 and batch_cnt!=0:
            emb_tensor = torch.cat(all_embs, dim=0)
            assert emb_tensor.shape[0] == sum(all_lengths) and len(all_doc_ids) == len(all_lengths)
            emb_save = {
                "doc_ids": all_doc_ids,
                "embs": emb_tensor,
                "lengths": all_lengths
            }
            torch.save(emb_save, f"{emb_filepate}emb_{save_idx}.pt")
            print(f"Saved final: {emb_filepate}emb_{save_idx}.pt")
            save_idx += 1
            all_embs, all_lengths, all_doc_ids = [], [], []

    if batch or all_embs:
        emb_tensor = torch.cat(all_embs, dim=0)
        assert emb_tensor.shape[0] == sum(all_lengths) and len(all_doc_ids) == len(all_lengths)
        emb_save = {
            "doc_ids": all_doc_ids,
            "embs": emb_tensor,
            "lengths": all_lengths
        }
        torch.save(emb_save, f"{emb_filepate}emb_{save_idx}.pt")
        print(f"Saved final: {emb_filepate}emb_{save_idx}.pt")
        save_idx += 1
        all_embs, all_lengths, all_doc_ids = [], [], []
    return doc_cnt