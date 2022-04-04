import attention
import torch
import torch.nn as nn
import torch.nn.functional as F

class attLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_size, dropout,use_bn, attn_head,
                 attn_size, activation="ReLU"):
        super(attLSTM,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.attn_head = attn_head
        self.attn_size = attn_size
        self.dropout = dropout
        self.use_bn = use_bn
        self.activation = getattr(nn, activation)()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lstm = nn.LSTM(input_size = self.input_dim, hidden_size = self.hidden_dim, 
                            num_layers = self.num_layers, bidirectional = True)
        self.attention = attention.Attention(self.batch_size, self.attn_head, self.attn_size, self.hidden_dim,
                                             self.hidden_dim, self.dropout)
        self.init_hidden_ = self.init_hidden()
        self.input_linear = nn.Linear(17,64)
        self.last_linear = nn.Linear(256,64)
        self.sigmoid =  nn.Sigmoid()

        
    def init_hidden(self):
        h = torch.empty(2*self.num_layers, self.batch_size, self.hidden_dim).to(self.device)
        c = torch.empty(2*self.num_layers, self.batch_size, self.hidden_dim).to(self.device)
        return (nn.init.xavier_normal_(h),
                nn.init.xavier_normal_(c))

    def forward(self, x):
        # self.hidden 각각의 layer의 모든 hidden state 를 갖고있음
        ## LSTM의 hidden state에는 tuple로 cell state포함, 0번째는 hidden state tensor, 1번째는 cell state
        # input dimension은 (Batch, Time_step, Feature dimension) 순이다. (batch_first=True)
        ## lstm_out : 각 time step에서의 lstm 모델의 output 값
        ## lstm_out[-1] : 맨마지막의 아웃풋 값으로 그 다음을 예측
        '''
        x : [64, 10, 17]
        lstm_input : [10, 128, 64]
        lstm_output : [10, 128, 128]  # 마지막 항 concat(64,64) <- bidirectional
        attn_applied : [128, 128]
        '''
        # print(x.size()) # torch.Size([64, 10, 17])

        x = self.input_linear(x.float().to(self.device))
        lstm_input = x.transpose(0,1)
        #print(lstm_input.size()) # torch.Size([10, 64, 17])
        lstm_out, self.hidden = self.lstm(lstm_input, self.init_hidden_)

        attn_applied, attn_weights = self.attention(lstm_out, lstm_out)
        # print(attn_applied.size()) torch.Size([64, 34])
        es = torch.cat([attn_applied, lstm_out[-1]], dim=1) # es : [128,256]
        # print(es.size()) # torch.Size([64, 68])
        yhat = self.last_linear(es)
        #yhat = self.sigmoid(lin_es)
        
        return yhat, attn_weights, attn_applied