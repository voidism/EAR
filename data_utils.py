from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import torch
from functools import partial
import time
from transformers import AutoTokenizer
import random
import pickle
import copy
import tqdm
import logging
import tqdm
from multiprocessing import Pool
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

def to_cuda(batch, gpuid):
    for n in batch:
        if n != "data" and n != "invert_index":
            batch[n] = batch[n].to(gpuid)


def collate_mp(batch, pad_token_id, is_test=False):
    def bert_pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(len(x) for x in X)
        result = []
        for x in X:
            if len(x) < max_len:
                x.extend([pad_token_id] * (max_len - len(x)))
            result.append(x)
        return torch.LongTensor(result)
    if len(batch) == 0:
        input_ids = bert_pad(batch[0]["input_ids"])
        ranks = torch.FloatTensor(batch[0]["ranks"])
        if "ctx_ids" in batch[0]:
            ctx_ids = bert_pad(batch[0]["ctx_ids"])
            result = {
                "input_ids": input_ids, 
                "ranks": ranks,
                "ctx_ids": ctx_ids
            }
        elif "invert_index" in batch[0]:
            result = {
                "input_ids": input_ids, 
                "ranks": ranks,
                "invert_index": batch[0]["invert_index"]
            }
        else:
            result = {
                "input_ids": input_ids, 
                "ranks": ranks
            }
        return result
    else:
        joint_input_ids = []
        joint_ranks = []
        chuck_sizes = []
        joint_ctx_ids = []
        for b in batch:
            joint_input_ids.extend(b["input_ids"])
            joint_ranks.extend(b["ranks"])
            chuck_sizes.append(len(b["input_ids"]))
            if "ctx_ids" in b:
                joint_ctx_ids.extend(b["ctx_ids"])
        input_ids = bert_pad(joint_input_ids)
        ranks = torch.FloatTensor(joint_ranks)
        chuck_sizes = torch.LongTensor(chuck_sizes)
        if len(joint_ctx_ids) > 0:
            ctx_ids = bert_pad(joint_ctx_ids)
            result = {
                "input_ids": input_ids, 
                "ranks": ranks,
                "chuck_sizes": chuck_sizes,
                "ctx_ids": ctx_ids
            }
        elif "invert_index" in batch[0]:
            result = {
                "input_ids": input_ids, 
                "ranks": ranks,
                "chuck_sizes": chuck_sizes,
                "invert_index": [b["invert_index"] for b in batch]
            }
        else:
            result = {
                "input_ids": input_ids, 
                "ranks": ranks,
                "chuck_sizes": chuck_sizes
            }
        return result


class ReRankingDataset(Dataset):
    def __init__(self, fdir, model_type, maxlen=64, is_test=False, total_len=512, is_sorted=False, maxnum=-1, null_rank=101, task_type="", org_query=False, dedup=False, rerank_size=0):
        """ data format: article, abstract, [(candidiate_i, score_i)] """
        cache_dir = f"cache-{fdir}-{model_type}-nullrank-101-maxlen-512-tasktype-{task_type}.pkl"
        if os.path.exists(cache_dir):
            print(f"Loading data from {cache_dir}")
            self.data = pickle.load(open(cache_dir, 'rb'))
            print(f"Finished loading data from {cache_dir}")
            self.cached = True
        else:
            self.data = json.load(open(fdir))
            self.cached = False
            self.tok = AutoTokenizer.from_pretrained(model_type, verbose=False)
            self.index2key = sorted(list(self.data.keys()), key=lambda x:int(x.split('-')[-1]))
            self.pad_token_id = self.tok.pad_token_id
            self.cls_token_id = self.tok.cls_token_id
            self.sep_token_id = self.tok.sep_token_id
            if not is_sorted and not is_test:
                for k in self.data:
                    self.data[k] = sorted(self.data[k], key=lambda x:x[1] if x[1] is not None else self.null_rank)
        self.num = len(self.data)
        self.maxlen = maxlen
        self.is_test = is_test
        self.total_len = total_len
        self.sorted = is_sorted
        self.maxnum = maxnum
        self.null_rank = null_rank
        self.task_type = task_type
        self.org_query = org_query
        reduced_len = []
        self.invert_index = {}
        self.reduced = False
        if rerank_size > 0:
            org_size = 0
            for key in self.data:
                if org_size == 0:
                    org_size = len(self.data[key])
                di = {}
                to_keep = []
                gen_times = 0
                #self.data[key] = self.data[key][:rerank_size]
                for i in range(len(self.data[key])):
                    if self.data[key][i][0] not in di:
                        di[self.data[key][i][0]] = 0
                        to_keep.append(i)
                        if len(to_keep) == rerank_size:
                            gen_times = i + 1
                            break
                self.data[key] = [self.data[key][i] for i in to_keep]
                self.invert_index[key] = {new_id:old_id for new_id, old_id in enumerate(to_keep)}
                reduced_len.append(gen_times)
            mean_len = sum(reduced_len)/len(reduced_len)
            std_len = (sum([((x - mean_len) ** 2) for x in reduced_len]) / len(reduced_len)) ** 0.5
            print(f"Finish reducing rerank_size from {org_size} to {rerank_size}")
            print("# of generations need to be performed per query:")
            print(f"Average #: {mean_len}, STD: {std_len}")
            self.reduced = True
        elif dedup:
            for key in self.data:
                di = {}
                to_keep = []
                for i in range(len(self.data[key])):
                    if self.data[key][i][0] not in di:
                        di[self.data[key][i][0]] = 0
                        to_keep.append(i)
                self.data[key] = [self.data[key][i] for i in to_keep]
                self.invert_index[key] = {new_id:old_id for new_id, old_id in enumerate(to_keep)}
                reduced_len.append(len(to_keep))
            mean_len = sum(reduced_len)/len(reduced_len)
            std_len = (sum([((x - mean_len) ** 2) for x in reduced_len]) / len(reduced_len)) ** 0.5
            print("Finish deduplication - remaining # of examples per query:")
            print(f"Average #: {mean_len}, STD: {std_len}")
            self.reduced = True

    def __len__(self):
        return self.num

    def bert_encode(self, x, max_len=64):
        segs = x.split(" ? ")
        q = segs[0]
        e = ' ? '.join(segs[1:])
        q_ids = self.tok.encode(q, add_special_tokens=False)
        e_ids = self.tok.encode(e, add_special_tokens=False)
        ids = [self.cls_token_id]
        if max_len > 0:
            ids.extend(q_ids[:max_len - len(ids) - 3])
            ids.append(self.sep_token_id)
            ids.extend(e_ids[:max_len - len(ids) - 3])
        else:
            ids.extend(q_ids[:self.total_len - len(ids) - 3])
            ids.append(self.sep_token_id)
            ids.extend(e_ids[:self.total_len - len(ids) - 3])
        ids.append(self.sep_token_id)
        return ids

    def bert_encode_wtop1(self, x, t, c, max_len=64):
        segs = x.split(" ? ")
        q = segs[0]
        e = ' ? '.join(segs[1:])
        q_ids = self.tok.encode(q, add_special_tokens=False)
        e_ids = self.tok.encode(e, add_special_tokens=False)
        t_ids = self.tok.encode(t, add_special_tokens=False)
        c_ids = self.tok.encode(c, add_special_tokens=False)
        n_sep = 5
        ids = [self.cls_token_id]
        if max_len > 0:
            ids.extend(q_ids[:max_len - len(ids) - n_sep])
            ids.append(self.sep_token_id)
            ids.extend(e_ids[:max_len - len(ids) - n_sep])
            ids.append(self.sep_token_id)
            ids.extend(t_ids[:max_len - len(ids) - n_sep])
            ids.append(self.sep_token_id)
            ids.extend(c_ids[:max_len - len(ids) - n_sep])
        else:
            ids.extend(q_ids[:self.total_len - len(ids) - n_sep])
            ids.append(self.sep_token_id)
            ids.extend(e_ids[:self.total_len - len(ids) - n_sep])
            ids.append(self.sep_token_id)
            ids.extend(t_ids[:self.total_len - len(ids) - n_sep])
            ids.append(self.sep_token_id)
            ids.extend(c_ids[:self.total_len - len(ids) - n_sep])
        ids.append(self.sep_token_id)
        return ids

    def bert_encode_contextual_wtop1(self, x, t, c, max_len=64):
        segs = x.split(" ? ")
        q = segs[0]
        e = ' ? '.join(segs[1:])
        q_ids = self.tok.encode(q, add_special_tokens=False)
        e_ids = self.tok.encode(e, add_special_tokens=False)
        t_ids = self.tok.encode(t, add_special_tokens=False)
        c_ids = self.tok.encode(c, add_special_tokens=False)
        n_sep = 5
        max_len = max_len if max_len > 0 else self.total_len

        ids = [self.cls_token_id]
        ids.extend(q_ids[:max_len - len(ids) - n_sep])
        ids.append(self.sep_token_id)
        ids.extend(e_ids[:max_len - len(ids) - n_sep])
        ids.append(self.sep_token_id)

        ctx_ids = [self.cls_token_id]
        ctx_ids.extend(t_ids[:max_len - len(ids) - n_sep])
        ctx_ids.append(self.sep_token_id)
        ctx_ids.extend(c_ids[:max_len - len(ids) - n_sep])
        ctx_ids.append(self.sep_token_id)
        return ids, ctx_ids

    def bert_encode_title(self, x, ts, max_len=64):
        segs = x.split(" ? ")
        q = segs[0]
        e = ' ? '.join(segs[1:])
        q_ids = self.tok.encode(q, add_special_tokens=False) + [self.sep_token_id]
        e_ids = self.tok.encode(e, add_special_tokens=False) + [self.sep_token_id]
        ts_ids = [(self.tok.encode(t, add_special_tokens=False) + [self.sep_token_id]) for t in ts]
        n_sep = 1
        ids = [self.cls_token_id]
        max_len = max_len if max_len > 0 else self.total_len
        ids.extend(q_ids[:max_len - len(ids)])
        ids.extend(e_ids[:max_len - len(ids)])
        for t_ids in ts_ids:
            ids.extend(t_ids[:max_len - len(ids)])
            if max_len == len(ids):
                break
        return ids

    def __getitem__(self, idx):
        if self.cached:
            item = self.data[idx]
            item['input_ids'] = [x[:self.maxlen] for x in item['input_ids']]
            if self.null_rank != 101:
                item['ranks'] = [(x if x != 101 else self.null_rank) for x in item['ranks']]
            return item
        key = self.index2key[idx]
        if self.task_type == "":
            if not self.org_query:
                input_ids = [self.bert_encode(x[0], self.maxlen) for x in self.data[key]]
            else:
                org_q = self.data[key][0][0].split(" ? ")[0]
                input_ids = [self.bert_encode(org_q, self.maxlen)] + [self.bert_encode(x[0], self.maxlen) for x in self.data[key]]
        elif self.task_type == "wtop1":
            input_ids = [self.bert_encode_wtop1(x[0], x[2], x[3], self.maxlen) for x in self.data[key]]
        elif self.task_type == "contextual_wtop1":
            inputs = [self.bert_encode_contextual_wtop1(x[0], x[2], x[3], self.maxlen) for x in self.data[key]]
            org_q = self.data[key][0][0].split(" ? ")[0]
            input_ids = [self.bert_encode(org_q, self.maxlen)] + [x[0] for x in inputs]
            ctx_ids = [x[1] for x in inputs]
        elif self.task_type == "title":
            input_ids = [self.bert_encode_title(x[0], x[2:], self.maxlen) for x in self.data[key]]
        else:
            raise ValueError("task_type not supported: %s" % self.task_type)
        ranks = [(x[1] if x[1] is not None else self.null_rank) for x in self.data[key]]
        if self.task_type == "contextual_wtop1":
            result = {
                "input_ids": input_ids,
                "ctx_ids": ctx_ids,
                "ranks": ranks
            }
        elif self.reduced:
            result = {
                "input_ids": input_ids,
                "ranks": ranks,
                "invert_index": self.invert_index[key]
                }
        else:
            result = {
                "input_ids": input_ids,
                "ranks": ranks
                }
        return result
