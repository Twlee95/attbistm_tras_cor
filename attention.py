# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/7 9:22
"""
"""
Compute 'Scaled Dot Product Attention

References:
    https://github.com/codertimo/BERT-pytorch/blob/d10dc4f9d5a6f2ca74380f62039526eb7277c671/bert_pytorch/model/attention/single.py#L8
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class Align(nn.Module):  
    def __init__(self, batch_size, attn_size, hid_dim):
        super(Align,self).__init__()

        self.attn_size = attn_size
        self.linear = nn.Linear(hid_dim+hid_dim,hid_dim+hid_dim)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.v = nn.init.xavier_normal_(torch.empty(batch_size, hid_dim+hid_dim, 1,requires_grad=True)).to(self.device)

## query : t 시점의 디코더 셀 에서의 은닉상태
## keys : 모든 시점의 인코더 셀의 은닉상태
## Value : 모든 시점의 인코더 셀의 은닉 상태
    def forward(self, key, value, mask=None, dropout=None):
        '''
        input_key : [10, 128, 128]  [seq_len, batch, feature] -> [128, 10, 1]
        input_value : [10, 128, 128]  [seq_len, batch, feature]
        att_scores : [128, 10, 10]
        
        '''
        scores = torch.matmul(torch.tanh(self.linear(key.transpose(0,1))),
                           self.v)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        p_attn = F.softmax(scores, dim=1)  # ([128, 10, 1]) [batch, seq_len, feature]
        
        if dropout is not None:
            p_attn = dropout(p_attn)
            
        value = value.transpose(0,1) # [batch, seq_len, feature] [128, 10, 128])

        x = value * p_attn # [128, 10, 128]) broadcasting
        x = torch.sum(x, axis=1) # [128, 10]
  
        return x , p_attn 

class Attention(nn.Module):
    """
    Take in model size and number of heads.
    general attention

    Args:
        query, key, value, mask. shape like (B, S, N)
    Returns:
        attention_value, (B, query_lens, N)
        attention_weight, (B, Head, query_lens, values_lens)
    References:
        https://github.com/codertimo/BERT-pytorch/blob/d10dc4f9d5a6f2ca74380f62039526eb7277c671/bert_pytorch/model/attention/single.py#L8
    """

    ## query : t 시점의 디코더 셀 에서의 은닉상태
    ## keys : 모든 시점의 인코더 셀의 은닉상태
    ## Value : 모든 시점의 인코더 셀의 은닉 상태
    def __init__(self, batch_size, heads, attn_size, key_size, value_size, dropout):
        super(Attention,self).__init__()
        assert attn_size % heads == 0

        # We assume d_v always equals d_k
        self.d_k = attn_size
        #self.linear_layers = nn.ModuleList([nn.Linear(s, attn_size) for s in [key_size, value_size]])
        self.align = Align(batch_size, attn_size, key_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, key, value, mask=None):
        """
        inputs shape (B, S, N)
        
        query, key, value = [torch.tanh(l(x)).view(batch_size, -1, self.d_k).transpose(1, 2)
                           for l, x in zip(self.linear_layers, (query, key, value))]
        이경우 nn.Linear(query_size, attn_size) 와 query 가 맨 첫번째로  .view 함수에 들어감
        
        """
        batch_size = key.size(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k

        x, attn = self.align(key, value, mask=mask, dropout=self.dropout)
        # torch.Size([128, 10, 1])
        
        x = x.contiguous()
        # (B, S, N), (B, H, S_q, S_k)
        return x, attn