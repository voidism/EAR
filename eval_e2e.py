import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import pickle
import time
import numpy as np
import os
import json
import random
from transformers import AutoModel, AutoTokenizer
from utils import Recorder
from data_utils import to_cuda, collate_mp, ReRankingDataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
from model import RankingLoss, ReRanker, ContextualReRanker, ContextualTop1ReRanker
import tqdm
import math
import logging
import wandb
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.ERROR)


def base_setting(args):
    args.batch_size = getattr(args, 'batch_size', 1)
    args.epoch = getattr(args, 'epoch', 5)
    args.report_freq = getattr(args, "report_freq", 100)
    args.accumulate_step = getattr(args, "accumulate_step", 12)
    args.margin = getattr(args, "margin", 0.01)
    args.gold_margin = getattr(args, "gold_margin", 0)
    args.model_type = getattr(args, "model_type", 'roberta-base')
    args.warmup_steps = getattr(args, "warmup_steps", 10000)
    args.grad_norm = getattr(args, "grad_norm", 0)
    args.seed = getattr(args, "seed", 970903)
    args.no_gold = getattr(args, "no_gold", False)
    args.pretrained = getattr(args, "pretrained", None)
    args.max_lr = getattr(args, "max_lr", 2e-3)
    args.scale = getattr(args, "scale", 1)
    args.train_file = getattr(args, "train_file", "test_set.json")
    args.valid_file = getattr(args, "valid_file", "test_set.json")
    args.max_len = getattr(args, "max_len", 64)
    args.max_num = getattr(args, "max_num", 16)
    args.cand_weight = getattr(args, "cand_weight", 1)
    args.gold_weight = getattr(args, "gold_weight", 1)


def evaluation(args):
    # load data
    base_setting(args)
    tok = AutoTokenizer.from_pretrained(args.model_type)
    collate_fn = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=True)
    test_set = ReRankingDataset(args.valid_file, args.model_type, is_test=True, maxlen=args.max_len, is_sorted=False, maxnum=args.max_num, task_type=args.task_type, org_query=(args.contextual or args.contextual_wtop1), dedup=args.dedup, rerank_size=args.rerank_size)
    dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    if args.contextual_wtop1:
        scorer = ContextualTop1ReRanker(model_path, tok.pad_token_id, args.feature_extractor_q, args.feature_extractor_c, post_processing=args.post_processing)
    elif args.contextual:
        scorer = ContextualReRanker(model_path, tok.pad_token_id, args.feature_extractor, post_processing=args.post_processing)
    else:
        scorer = ReRanker(model_path, tok.pad_token_id)
    if args.cuda:
        scorer = scorer.cuda()
    # scorer.load_state_dict(torch.load(os.path.join(args.output_dir, args.model_pt), map_location=f'cuda:{args.gpuid[0]}'))
    state_dict = torch.load(args.model_pt, map_location=f'cuda:{args.gpuid[0]}')
    # if args.contextual:
    #     new_state_dict = {}
    #     for key in state_dict:
    #         if not key.startswith("feature_extractor"):
    #             new_state_dict[key] = state_dict[key]
    #     state_dict = new_state_dict
    scorer.load_state_dict(state_dict)
    scorer.eval()
    model_name = args.model_pt.split("/")[0]

    # loss = 0
    cnt = 0
    acc = 0
    best_rank = 0
    mean_rank = 0
    pred_rank = 0
    top1s = []
    top1s_gt = []
    pred_rank_list = []
    time_st = time.time()
    with torch.no_grad():
        for (i, batch) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing"):
            if args.cuda:
                to_cuda(batch, args.gpuid[0])
            ranks = batch["ranks"]
            #random_indices = torch.randperm(ranks.size(0)).to(ranks.device)
            #zero_lead = torch.zeros(1, dtype=random_indices.dtype).to(ranks.device)
            #shuffler = torch.cat([zero_lead, random_indices+1], dim=0).to(ranks.device)
            #unshuffler = torch.argsort(random_indices).to(ranks.device)
            #ranks = ranks[random_indices]
            #batch["input_ids"] = batch["input_ids"][shuffler]
            if args.fp16:
                with torch.cuda.amp.autocast():
                    output = scorer(batch["input_ids"], random_order=False) if not args.contextual_wtop1 else scorer(batch["input_ids"], batch["ctx_ids"])
            else:
                output = scorer(batch["input_ids"], random_order=False) if not args.contextual_wtop1 else scorer(batch["input_ids"], batch["ctx_ids"])
            if 'chuck_sizes' not in batch or len(batch['chuck_sizes']) == 1:
                all_ranks, all_output = [ranks], [output]
            else:
                pivot = 0
                all_ranks, all_output = [], []
                for chuck_size in batch["chuck_sizes"]:
                    all_ranks.append(ranks[pivot:pivot+chuck_size])
                    all_output.append(output[pivot:pivot+chuck_size])
                    pivot += chuck_size
            all_invert_index = batch["invert_index"] if "invert_index" in batch else [None] * len(all_ranks)

            for ranks, output, invert_index in zip(all_ranks, all_output, all_invert_index):
                top1 = torch.argmin(output, dim=0)
                top1_gt = int(torch.argmin(ranks, dim=0))
                min_rank = torch.min(ranks, dim=0)[0]
                top1_ranks = (min_rank == ranks).nonzero().squeeze(1)
                best_rank += int(ranks[int(top1_gt)])
                mean_rank += torch.mean(ranks, dim=0)
                pred_rank += int(ranks[int(top1)])
                pred_rank_list.append(int(ranks[int(top1)]))
                if invert_index is not None:
                    top1s.append(invert_index[int(top1)])
                    top1s_gt.append(invert_index[int(top1_gt)])
                else:
                    top1s.append(int(top1))
                    top1s_gt.append(int(top1_gt))
                # print(f"top1: {int(top1)} with rank={ranks[int(top1)]}, gt: {top1_gt} with rank={ranks[top1_gt]}")
                if top1 in top1_ranks:
                    acc += 1
                # loss += RankingLoss(output, batch["ranks"], args.margin, args.loss_type)
                cnt += 1
    time_ed = time.time()
    print(f"Time: {time_ed - time_st:.6f}s")
    print(f"Time per query: {(time_ed - time_st) / cnt:.6f}s")
    print(f"accuracy: {acc / cnt:.6f} avg best rank: {best_rank / cnt:.6f} avg mean rank: {mean_rank / cnt:.6f} avg pred rank: {pred_rank / cnt:.6f}")
    pred_rank_list = np.array(pred_rank_list)
    topk = {}
    to_print = ""
    for k in [1, 5, 10, 20, 50, 100]:
        topk[k] = float(np.mean(pred_rank_list <= k))
        print("Top-%d:"%k, topk[k])
        to_print += str(topk[k]) + ', '
    print(to_print[:-2])
    
    return top1s, top1s_gt


def test(dataloader, scorer, args, gpuid, target):
    scorer.eval()
    if args.cuda:
        scorer.cuda(gpuid)
    loss = 0
    cnt = 0
    acc = 0
    best_rank = 0
    mean_rank = 0
    pred_rank = 0
    pred_rank_list = []
    best_rank_list = []
    with torch.no_grad():
        for (i, batch) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing"):
            if args.cuda:
                to_cuda(batch, args.gpuid[0])
            ranks = batch["ranks"]
            if args.fp16:
                with torch.cuda.amp.autocast():
                    output = scorer(batch["input_ids"]) if not args.contextual_wtop1 else scorer(batch["input_ids"], batch["ctx_ids"])
            else:
                output = scorer(batch["input_ids"]) if not args.contextual_wtop1 else scorer(batch["input_ids"], batch["ctx_ids"])
            if 'chuck_sizes' not in batch or len(batch['chuck_sizes']) == 1:
                all_ranks, all_output = [ranks], [output]
            else:
                pivot = 0
                all_ranks, all_output = [], []
                for chuck_size in batch["chuck_sizes"]:
                    all_ranks.append(ranks[pivot:pivot+chuck_size])
                    all_output.append(output[pivot:pivot+chuck_size])
                    pivot += chuck_size

            for ranks, output in zip(all_ranks, all_output):
                top1 = torch.argmin(output, dim=0)
                min_rank = torch.min(ranks, dim=0)[0]
                top1_ranks = (min_rank == ranks).nonzero().squeeze(1)
                best_rank += int(min_rank)
                mean_rank += torch.mean(ranks, dim=0)
                pred_rank += int(ranks[int(top1)])
                pred_rank_list.append(int(ranks[int(top1)]))
                best_rank_list.append(int(min_rank))
                if top1 in top1_ranks:
                    acc += 1
                loss += RankingLoss(output, ranks, args.margin, args.loss_type)
                cnt += 1
    print(f"\naccuracy: {acc / cnt:.6f} loss: {loss / cnt:.6f} avg best rank: {best_rank / cnt:.6f} avg mean rank: {mean_rank / cnt:.6f} avg pred rank: {pred_rank / cnt:.6f}")
    pred_rank_list = np.array(pred_rank_list)
    best_rank_list = np.array(best_rank_list)
    topk = {}
    for k in [1, 5, 10, 20, 50, 100]:
        topk[f"eval-{target}/top-%d-acc"%k] = float(np.mean(pred_rank_list <= k))
        print("Target %s Top-%d:"%(target, k), topk[f"eval-{target}/top-%d-acc"%k])
    for k in [1, 5, 10, 20, 50, 100]:
        print("Target %s Upper bound - Top-%d:"%(target, k), float(np.mean(best_rank_list <= k)))
    return loss / cnt, acc / cnt, best_rank / cnt, mean_rank / cnt, pred_rank / cnt, topk

def work(x):
    cache, train_set, x = x
    cache.append(train_set[x])


def run(rank, args):
    base_setting(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    config = vars(args)
    if args.wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config
        )
    gpuid = args.gpuid[rank]
    is_master = rank == 0
    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)
    if is_master:
        id = len(os.listdir(args.output_dir))
        recorder = Recorder(id, args.log, name=args.wandb_name)
    tok = AutoTokenizer.from_pretrained(args.model_type)
    collate_fn = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=False)
    collate_fn_val = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=True)
    train_set = ReRankingDataset(args.train_file, args.model_type, is_test=True, maxlen=args.max_len, maxnum=args.max_num, null_rank=args.null_rank, task_type=args.task_type, org_query=(args.contextual or args.contextual_wtop1))
    if False and args.task_type == "title":
        import multiprocessing as mp
        n_workers = 60
        pool = mp.Pool(n_workers)
        print("Multi Processing")
        cache = []
        for _ in tqdm.tqdm(pool.imap_unordered(work, [(cache, train_set, x) for x in range(len(train_set))]), total=len(train_set)):
            pass
        import pickle
        fw = open('train-title.pkl', 'wb')
        pickle.dump(cache, fw)
        #self.cache = pool.map(self.__getitem__, list(range(self.num)))
        #for i in tqdm.trange(self.num, desc="Saving the cache"):
        #    self.cache[i] = self.__getitem__(i)
        #self.cache_done = True
        #del self.data
        import pdb; pdb.set_trace()


    if args.valid_files is None:
        target, valid_file = args.valid_file.split(':')
        val_sets = [(target, ReRankingDataset(valid_file, args.model_type, is_test=True, maxlen=512, is_sorted=False, maxnum=args.max_num, task_type=args.task_type, org_query=(args.contextual or args.contextual_wtop1)))]
    else:
        val_sets = []
        for valid_file in args.valid_files.split(','):
            target, valid_file = valid_file.split(':')
            val_sets.append((target, ReRankingDataset(valid_file, args.model_type, is_test=True, maxlen=512, is_sorted=False, maxnum=args.max_num, task_type=args.task_type, org_query=(args.contextual or args.contextual_wtop1))))
    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
    	 train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn, sampler=train_sampler)
        val_samplers = [torch.utils.data.distributed.DistributedSampler(
    	 val_set, num_replicas=world_size, rank=rank) for val_set in val_sets]
        val_dataloaders = [(target, DataLoader(val_set, batch_size=8, shuffle=False, num_workers=8, collate_fn=collate_fn_val, sampler=val_samplers[i])) for i, (target, val_set) in enumerate(val_sets)]
    else:
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)
        val_dataloaders = [(target, DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn_val)) for target, val_set in val_sets]
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    if args.contextual_wtop1:
        scorer = ContextualTop1ReRanker(model_path, tok.pad_token_id, args.feature_extractor_q, args.feature_extractor_c, post_processing=args.post_processing)
    elif args.contextual:
        scorer = ContextualReRanker(model_path, tok.pad_token_id, args.feature_extractor, post_processing=args.post_processing)
    else:
        scorer = ReRanker(model_path, tok.pad_token_id)
    # scorer = ReRanker(model_path, tok.pad_token_id) if not args.contextual else ContextualReRanker(model_path, tok.pad_token_id, args.feature_extractor)
    if len(args.model_pt) > 0:
        scorer.load_state_dict(torch.load(os.path.join(args.output_dir, args.model_pt), map_location=f'cuda:{gpuid}'))
    if args.cuda:
        if len(args.gpuid) == 1:
            scorer = scorer.cuda()
        else:
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            scorer = nn.parallel.DistributedDataParallel(scorer.to(gpuid), [gpuid], find_unused_parameters=True)
    scorer.train()
    init_lr = args.max_lr / args.warmup_steps
    s_optimizer = optim.Adam(scorer.parameters(), lr=init_lr)
    if is_master:
        recorder.write_config(args, [scorer], __file__)
    min_avg_ranks = {target:10000 for target, _ in val_sets}
    all_step_cnt = 0
    # start training
    for epoch in range(args.epoch):
        s_optimizer.zero_grad()
        step_cnt = 0
        sim_step = 0
        avg_loss = 0
        print(f"Epoch {epoch}")
        progress_bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Training epoch {}".format(epoch))
        eval_steps = len(dataloader)//args.eval_per_epoch
        did_first_eval = False
        for (i, batch) in progress_bar:
            pass
            if args.cuda:
                to_cuda(batch, gpuid)
            step_cnt += 1
            if args.fp16:
                with torch.cuda.amp.autocast():
                    output = scorer(batch["input_ids"]) if not args.contextual_wtop1 else scorer(batch["input_ids"], batch["ctx_ids"])
                    if args.batch_size == 1:
                        loss = args.scale * RankingLoss(output, batch["ranks"], args.margin, args.loss_type)
                    else:
                        pivot = 0
                        loss = 0.0
                        for chuck_size in batch["chuck_sizes"]:
                            loss += args.scale * RankingLoss(output[pivot:pivot+chuck_size], batch["ranks"][pivot:pivot+chuck_size], args.margin, args.loss_type)
                            pivot += chuck_size
                        loss = loss / len(batch["chuck_sizes"])
            else:
                output = scorer(batch["input_ids"]) if not args.contextual_wtop1 else scorer(batch["input_ids"], batch["ctx_ids"])
                if args.batch_size == 1:
                    loss = args.scale * RankingLoss(output, batch["ranks"], args.margin, args.loss_type)
                else:
                    pivot = 0
                    loss = 0.0
                    for chuck_size in batch["chuck_sizes"]:
                        loss += args.scale * RankingLoss(output[pivot:pivot+chuck_size], batch["ranks"][pivot:pivot+chuck_size], args.margin, args.loss_type)
                        pivot += chuck_size
                    loss = loss / len(batch["chuck_sizes"])
                loss = loss / args.accumulate_step
            avg_loss += loss.item()
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            progress_bar.set_postfix({"loss": loss.item()})
            if args.wandb:
                wandb.log({"train/loss": loss.item()})
            if step_cnt == args.accumulate_step:
                # optimize step      
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(scorer.parameters(), args.grad_norm)
                step_cnt = 0
                sim_step += 1
                all_step_cnt += 1
                lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))
                for param_group in s_optimizer.param_groups:
                    param_group['lr'] = lr
                if args.fp16:
                    scaler.step(s_optimizer)
                else:
                    s_optimizer.step()
                s_optimizer.zero_grad()
                scaler.update()
            if sim_step % args.report_freq == 0 and step_cnt == 0 and is_master:
                print("id: %d"%id)
                recorder.print("epoch: %d, batch: %d, avg loss: %.6f"%(epoch+1, sim_step, 
                 avg_loss / args.report_freq))
                recorder.print(f"learning rate: {lr:.6f}")
                recorder.plot("loss", {"loss": avg_loss / args.report_freq}, all_step_cnt)
                recorder.print()
                avg_loss = 0

            # if all_step_cnt % 1000 == 0: # and step_cnt == 0 all_step_cnt != 0 and 
            if ((i % eval_steps == 0 and i>0) or i == len(dataloader)-1) or (not did_first_eval and not args.skip_eval): # and step_cnt == 0 all_step_cnt != 0 and 
                for target, val_dataloader in val_dataloaders:
                    loss, acc, best_rank, mean_rank, pred_rank, topk = test(val_dataloader, scorer, args, gpuid, target)
                    did_first_eval = True
                    if args.wandb:
                        d = {f"eval-{target}/loss": loss,
                             f"eval-{target}/acc": acc,
                             f"eval-{target}/best_rank": best_rank,
                             f"eval-{target}/mean_rank": mean_rank,
                             f"eval-{target}/pred_rank": pred_rank}
                        for k in topk:
                            d[k] = topk[k]
                        wandb.log(d) 
                    if pred_rank < min_avg_ranks[target] and is_master:
                        min_avg_ranks[target] = pred_rank
                        if is_mp:
                            recorder.save(scorer.module, f"scorer-{target}-best.bin")
                        else:
                            recorder.save(scorer, f"scorer-{target}-best.bin")
                        recorder.print("eval target: %s"%target)
                        recorder.print("best - epoch: %d, batch: %d"%(epoch, i / args.accumulate_step))
                        recorder.print("best - loss: %.6f, acc: %.6f, best rank: %.6f, mean rank: %.6f, pred rank: %.6f"%(loss, acc, best_rank, mean_rank, pred_rank))
                recorder.save(scorer, "scorer-last.bin")
                recorder.save(s_optimizer, "optimizer.bin")
                # if is_master:
                #     recorder.print("val rouge: %.6f"%(1 - loss))
    recorder.save(scorer, "scorer-last.bin")
    torch.save(scorer.state_dict(), os.path.join(args.output_dir, args.wandb_name+"-scorer-last.bin"))

def main(args):
    # set env
    if len(args.gpuid) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'{args.port}'
        mp.spawn(run, args=(args,), nprocs=len(args.gpuid), join=True)
    else:
        run(0, args)

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Training Parameter')
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--wtop1", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--contextual", action="store_true")
    parser.add_argument("--contextual_wtop1", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default='ugar', type=str)
    parser.add_argument("--wandb_name", default='roberta-base', type=str)
    parser.add_argument("--model_type", default='roberta-base', type=str)
    parser.add_argument("--feature_extractor", default='facebook/dpr-question_encoder-single-nq-base', type=str)
    parser.add_argument("--feature_extractor_q", default='facebook/dpr-question_encoder-single-nq-base', type=str)
    parser.add_argument("--feature_extractor_c", default='facebook/dpr-ctx_encoder-single-nq-base', type=str)
    parser.add_argument("--gpuid", nargs='+', type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--max_lr", type=float, default=2e-3)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--accumulate_step", type=int, default=12)
    parser.add_argument("--eval_per_epoch", type=int, default=70)
    parser.add_argument("-e", "--evaluate", action="store_true")
    parser.add_argument("-l", "--log", action="store_true")
    parser.add_argument("-p", "--port", type=int, default=12355)
    parser.add_argument("--model_pt", default="", type=str)
    parser.add_argument("--dataset", default="nq", type=str)
    parser.add_argument("--task_type", default="", type=str)
    parser.add_argument("--post_processing", default="diff", type=str)
    parser.add_argument("--encode_mode", default=None, type=str)
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--valid_file", default=None, type=str)
    parser.add_argument("--valid_files", default=None, type=str)
    parser.add_argument("--pretrained", default=None, type=str)
    parser.add_argument("--bm25_dir", default="output_t0_gen_bm25", type=str)
    parser.add_argument("--output_dir", default="./cache", type=str)
    parser.add_argument("--rerank_output", default="reranked_result.json", type=str)
    parser.add_argument("--loss_type", default='weight-divide', type=str)
    parser.add_argument("--null_rank", type=int, default=101)
    parser.add_argument("--skip_eval_first", action="store_true")
    parser.add_argument("--dedup", action="store_true")
    parser.add_argument("--rerank_size", type=int, default=0)
    args = parser.parse_args()
    if args.cuda is False:
        if args.evaluate:
            result = evaluation(args)
        else:
            main(args)
    else:
        if args.evaluate:
            with torch.cuda.device(args.gpuid[0]):
                result, result_gt = evaluation(args)
                if args.bm25_dir is not None:
                    reranked_result = []
                    reranked_result_gt = []
                    if args.n_workers <= 1:
                        for i, r in tqdm.tqdm(enumerate(result), total=len(result), desc="Reranking"):
                            # open the bm25 file and get the r-th item
                            with open(os.path.join(args.bm25_dir, f"{args.dataset}-test-{i}/results.json")) as f:
                                bm25_result = json.load(f)
                                reranked_result.append(bm25_result[r+1])
                                reranked_result_gt.append(bm25_result[result_gt[i]+1])
                    else:
                        def rerank(i, r):
                            with open(os.path.join(args.bm25_dir, f"{args.dataset}-test-{i}/results.json")) as f:
                                bm25_result = json.load(f)
                                return bm25_result[r+1]
                        pool = mp.Pool(args.n_workers)
                        reranked_result = pool.starmap(rerank, enumerate(result))
                        pool.close()
                        pool.join()

                        pool = mp.Pool(args.n_workers)
                        reranked_result_gt = pool.starmap(rerank, enumerate(result_gt))
                        pool.close()
                        pool.join()
                    # write to file
                    with open(args.rerank_output, "w") as f:
                        json.dump(reranked_result, f, indent=4)
                    with open(args.rerank_output.replace(".json", "_gt.json"), "w") as f:
                        json.dump(reranked_result_gt, f, indent=4)
        elif len(args.gpuid) == 1:    
            with torch.cuda.device(args.gpuid[0]):
                main(args)
        else:
            main(args)
