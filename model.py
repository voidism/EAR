# modified from https://github.com/maszhongming/MatchSum
import torch
from torch import nn
from transformers import AutoModel, BertModel, BertTokenizer, BertConfig, DPRContextEncoder, DPRQuestionEncoder


def RankingLoss(score, ranks, margin=1.0, loss_type="equal-divide", sorted=False):
    ones = torch.ones_like(score)
    # loss_func = torch.nn.MarginRankingLoss(0.0)
    indices = torch.argsort(ranks)
    ranks = ranks[indices]
    score = score[indices]
    TotalLoss = 0.0 #loss_func(score, score, ones)
    # candidate loss
    n = score.size(0)
    total = 0
    for i in range(1, n):
        pos_score = score[:-i]
        neg_score = score[i:]
        rank_diff = ranks[i:] - ranks[:-i]
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        loss = torch.max(pos_score - neg_score + margin * rank_diff, torch.zeros_like(pos_score))
        if loss_type in ["equal-divide", "equal-sum"]:
            TotalLoss += torch.sum(loss)
            total += loss.size(0)
        elif loss_type in ["weight-divide", "weight-sum"]:
            TotalLoss += torch.mean(loss)
            total += 1
        else:
            raise NotImplementedError
    if loss_type in ["equal-divide", "weight-divide"]:
        TotalLoss /= total
    return TotalLoss


class ReRanker(nn.Module):
    def __init__(self, encoder, pad_token_id):
        super(ReRanker, self).__init__()
        self.encoder = AutoModel.from_pretrained(encoder)
        self.scorer = nn.Linear(self.encoder.config.hidden_size, 1)
        self.pad_token_id = pad_token_id

    def forward(self, text_id, random_order=False):
        
        batch_size = text_id.size(0)
        
        input_mask = text_id != self.pad_token_id
        out = self.encoder(text_id, attention_mask=input_mask)[0]
        doc_emb = out[:, 0, :]
        scores = self.scorer(doc_emb).squeeze(-1)
        return scores
        

class ContextualReRanker(nn.Module):
    def __init__(self, encoder, pad_token_id, feature_extractor, post_processing='diff'):
        super(ContextualReRanker, self).__init__()
        self.feature_extractor = AutoModel.from_pretrained(feature_extractor)
        self.encoder = AutoModel.from_pretrained(encoder)
        self.scorer = nn.Linear(self.encoder.config.hidden_size, 1)
        self.pad_token_id = pad_token_id
        self.post_processing = post_processing
        self.feature_extractor.eval()

    def forward(self, text_id, random_order=False):        
        input_mask = text_id != self.pad_token_id
        with torch.no_grad():
            outputs = self.feature_extractor(text_id, attention_mask=input_mask)
            embs = outputs[0]
            if self.post_processing == 'diff':
                to_sub = torch.cat([torch.zeros(1, embs.shape[1]).to(embs.device), embs[:1,:].repeat(50, 1)], dim=0)
                embs = embs - to_sub
        if random_order:
            random_indices = torch.randperm(embs.size(0)-1).to(embs.device)
            zero_lead = torch.zeros(1, dtype=random_indices.dtype).to(embs.device)
            shuffler = torch.cat([zero_lead, random_indices+1], dim=0).to(embs.device)
            unshuffler = torch.argsort(random_indices).to(embs.device)
            embs = embs[shuffler]
        embs = embs.unsqueeze(0).detach()
        out = self.encoder(inputs_embeds=embs, attention_mask=None)
        doc_emb = out.last_hidden_state.squeeze(0)[1:, :]
        scores = self.scorer(doc_emb).squeeze(-1)
        if random_order:
            scores = scores[unshuffler]
        return scores
        
class ContextualTop1ReRanker(nn.Module):
    def __init__(self, model_path=None, pad_token_id=0, feature_extractor_q=None, feature_extractor_c=None, post_processing='diff'):
        super(ContextualTop1ReRanker, self).__init__()
        self.feature_extractor_q = DPRQuestionEncoder.from_pretrained(feature_extractor_q)
        self.feature_extractor_c = DPRContextEncoder.from_pretrained(feature_extractor_c)
        encoder_config = "ct_reranker.json"
        self.config = BertConfig.from_json_file(encoder_config)
        self.encoder = BertModel(self.config)
        self.scorer = nn.Linear(self.encoder.config.hidden_size, 1)
        self.pad_token_id = pad_token_id
        self.post_processing = post_processing
        self.feature_extractor_q.eval()
        self.feature_extractor_c.eval()

    def forward(self, text_id, top1_ctxs, random_order=False):        
        input_mask = text_id != self.pad_token_id
        top1_ctxs_mask = top1_ctxs != self.pad_token_id
        with torch.no_grad():
            outputs = self.feature_extractor_q(text_id, attention_mask=input_mask)
            ctx_outputs = self.feature_extractor_c(top1_ctxs, attention_mask=top1_ctxs_mask)
            q_embs = outputs[0]
            ctx_embs = ctx_outputs[0]
            embs = torch.cat([q_embs[:1,:].repeat(50, 1), q_embs[1:, :], ctx_embs], dim=1)
            # if self.post_processing == 'diff':
            #     to_sub = torch.cat([torch.zeros(1, embs.shape[1]).to(embs.device), embs[:1,:].repeat(50, 1)], dim=0)
            #     embs = embs - to_sub
        if random_order:
            shuffler = torch.randperm(embs.size(0)).to(embs.device)
            unshuffler = torch.argsort(shuffler).to(embs.device)
            embs = embs[shuffler]
        embs = embs.unsqueeze(0).detach()
        out = self.encoder(inputs_embeds=embs, attention_mask=None)
        doc_emb = out.last_hidden_state.squeeze(0)
        scores = self.scorer(doc_emb).squeeze(-1)
        if random_order:
            scores = scores[unshuffler]
        return scores
