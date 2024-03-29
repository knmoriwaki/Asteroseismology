import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchhk import transform_model
import torchbnn as bnn

def MyModel(args):

    n_feature_in = 1 if isinstance(args.input_id, int) else len(args.input_id)
    n_feature_out = 1 if isinstance(args.output_id, int) else len(args.output_id)

    if args.loss == "nllloss":
        last_act = nn.LogSoftmax(dim=1)
    else:
        args.output_dim = 1
        last_act = nn.Sigmoid() 

    if args.model == "RNN":
        model = RecurrentNet(n_feature_in=n_feature_in, n_feature_out=n_feature_out, seq_length_out=args.output_dim, hidden_dim=args.hidden_dim, n_layer=args.n_layer, last_act=last_act)

    elif args.model == "Dhanpal22":
        model = ConvNet(Conv1dBlock, n_feature_in=n_feature_in, n_feature_out=n_feature_out, seq_length=args.seq_length, seq_length_out=args.output_dim, hidden_dim=args.hidden_dim, n_layer=args.n_layer, r_drop=args.r_drop, batch_norm=args.batch_norm, last_act=last_act, nlayer_increase=args.nlayer_increase, additional_layer="lstm2")

    elif args.model == "MDN":
        model = MixtureDensityNetwork(Conv1dBlock, K_mdn=args.K_mdn, n_feature_in=n_feature_in, n_feature_out=n_feature_out, seq_length=args.seq_length, hidden_dim=args.hidden_dim, n_layer=args.n_layer, nlayer_increase=args.nlayer_increase, last_act=nn.Sigmoid(), additional_layer="lstm2")

    elif "CNN" in args.model:
        if args.seq_length_2 > 0:
            model = Conv2dNet(Conv2dBlock, n_feature_in=n_feature_in, n_feature_out=n_feature_out, width=args.seq_length, height=args.seq_length_2, seq_length_out=args.output_dim, hidden_dim=args.hidden_dim, n_layer=args.n_layer, r_drop=args.r_drop, batch_norm=args.batch_norm, last_act=last_act)

        else:
            if "LSTM" in args.model:
                additional_layer = "lstm"
            elif "attention" in args.model:
                additional_layer = "attention"
            else:
                additional_layer = "" 

            model = ConvNet(Conv1dBlock, n_feature_in=n_feature_in, n_feature_out=n_feature_out, seq_length=args.seq_length, seq_length_out=args.output_dim, hidden_dim=args.hidden_dim, n_layer=args.n_layer, r_drop=args.r_drop, batch_norm=args.batch_norm, last_act=last_act, nlayer_increase=args.nlayer_increase, additional_layer=additional_layer)

    elif args.model == "BNN":
        if args.loss == "nllloss":
            print("Error: The current version does not allow nllloss for BNN", file=sys.stderr)
            sys.exit(1)
        model = ConvNet(Conv1dBlock_w_Conv2d, n_feature_in=n_feature_in, n_feature_out=n_feature_out, seq_length=args.seq_length, seq_length_out=args.output_dim, hidden_dim=args.hidden_dim, n_layer=args.n_layer, r_drop=args.r_drop, batch_norm=args.batch_norm, last_act=last_act)
        transform_model(model, nn.Conv2d, bnn.BayesConv2d, 
            args={"prior_mu":0, "prior_sigma":0.1, "in_channels" : ".in_channels",
                  "out_channels" : ".out_channels", "kernel_size" : ".kernel_size",
                  "stride" : ".stride", "padding" : ".padding", "bias":".bias"}, 
            attrs={"weight_mu" : ".weight"})
        transform_model(model, nn.Linear, bnn.BayesLinear,
            args={"prior_mu":0, "prior_sigma":0.1, "in_features" : ".in_features",
                  "out_features" : ".out_features", "bias":".bias"}, 
            attrs={"weight_mu" : ".weight"})

    else:
        print("Error: unkonwn model", file=sys.stderr)
        sys.exit(1)
    return model

class Conv1dBlock(nn.Module):

    def __init__(self, nin=32, nout=32, kernel_size=3, stride=2, padding="same", bn=False, r_drop=0):
        super().__init__()

        self.bn = bn
        self.drop = True if r_drop > 0 else False

        self.conv = nn.Conv1d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding)
        if self.bn:
            self.batch_norm = nn.BatchNorm1d(nout)
        if self.drop:
            self.dropout = nn.Dropout(r_drop)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.batch_norm(x)
        if self.drop:
            x = self.dropout(x)
        x = self.act(x)
        return x

class Conv1dBlock_w_Conv2d(nn.Module):

    def __init__(self, nin=32, nout=32, kernel_size=3, stride=2, padding="same", bn=False, r_drop=0):
        super().__init__()

        self.bn = bn
        self.drop = True if r_drop > 0 else False

        self.conv = nn.Conv2d(nin, nout, kernel_size=(kernel_size,1), stride=(stride,1), padding=(padding,0))
        if self.bn:
            self.batch_norm = nn.BatchNorm2d(nout)
        if self.drop:
            self.drop = nn.Dropout(r_drop)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = torch.unsqueeze(x, dim=-1)
        x = self.conv(x)
        if self.bn:
            x = self.batch_norm(x)
        if self.drop:
            x = self.drop(x)
        x = self.act(x)
        x = torch.squeeze(x, dim=-1)
        return x

class Conv2dBlock(nn.Module):

    def __init__(self, nin=32, nout=32, kernel_size=[3,3], stride=[2,2], padding="same", bn=False, r_drop=0):
        super().__init__()

        self.bn = bn
        self.drop = True if r_drop > 0 else False

        self.conv = nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding)
        if self.bn:
            self.batch_norm = nn.BatchNorm2d(nout)
        if self.drop:
            self.dropout = nn.Dropout(r_drop)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.batch_norm(x)
        if self.drop:
            x = self.dropout(x)
        x = self.act(x)
        return x

class RecurrentNet(nn.Module):

    def __init__(self, n_feature_in=8, n_feature_out=1, seq_length_out=10, hidden_dim=32, n_layer=1, nonlinearity="tanh", r_drop=0, last_act=nn.LogSoftmax(dim=1)):
        super().__init__()

        self.seq_length_out = seq_length_out

        self.rnn = nn.RNN(n_feature_in, hidden_dim, n_layer, nonlinearity=nonlinearity, batch_first=True, dropout=r_drop, bidirectional=False) 
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, seq_length_out*n_feature_out)
        self.output_act = last_act

    def forward(self, x):
        batch_size = x.size(0)
        ## x: (batch, seq, n_feature_in)

        out, h_last = self.rnn(x)
        ## out: (batch, seq, hidden_dim)
        ## h_last: (n_layer, batch, hidden_dim)

        out = out[:,-1,:].contiguous().view(batch_size, -1)
        ## out: (batch, hidden_dim)

        out = self.linear1(out)
        out = self.linear2(out)
        out = self.output_act(out)
        ## out: (batch, seq_length_out*n_feature_out)
        
        if seq_length_out != 1:
            out = out.reshape(batch_size, self.seq_length_out, -1)
            ## out: (batch, seq_length_out, n_feature_out)

        return out

class ConvNet(nn.Module):

    def __init__(self, block, n_feature_in=8, n_feature_out=1, seq_length=10, seq_length_out=10, hidden_dim=32, n_layer=4, kernel_size=3, r_drop=0, batch_norm=False, last_act=nn.LogSoftmax(dim=1), nlayer_increase=3, additional_layer="" ):
        super().__init__()

        self.seq_length_out = seq_length_out

        padding = int( kernel_size / 2 )

        input_dims = [ n_feature_in ] + [ hidden_dim * min(2**i, 2**(nlayer_increase-1)) for i in range(n_layer-1) ]
        output_dims = [ hidden_dim * min(2**i, 2**(nlayer_increase-1)) for i in range(n_layer) ]

        if n_layer == 1:
            dropout_rates = [ r_drop ]
        else:
            dropout_rates = [0, 0] + [ r_drop for i in range(n_layer-2) ] 

        self.blocks = nn.ModuleList([
            block(nin=i, nout=j, stride=2, kernel_size=kernel_size, padding=padding, bn=batch_norm, r_drop=r)
            for i, j, r in zip(input_dims, output_dims, dropout_rates)
            ])

        ### e.g., for seq_length = 10 with 
        ### (input_dim, 10) -> (hidden_dim*2, 5) -> (hidden_dim*4, 3) -> (hidden_dim*8, 2) 
        seq_length_tmp = seq_length
        for i in range(n_layer): seq_length_tmp = int( ( seq_length_tmp + 1 ) / 2 )

        self.additional_layer = additional_layer
        if "lstm" in self.additional_layer:
            self.hidden_size = 256
            self.n_layer_lstm = 2 if "2" in self.additional_layer else 1
            self.lstm = nn.LSTM(input_size=output_dims[-1], hidden_size=self.hidden_size, num_layers=self.n_layer_lstm, batch_first=True, dropout=r_drop)
            final_dim = seq_length_tmp * self.hidden_size
        elif self.additional_layer == "attention":
            self.attention = nn.MultiheadAttention(embed_dim=output_dims[-1], num_heads=8, batch_first=True)
            final_dim = seq_length_tmp * output_dims[-1] 
        else:
            final_dim = seq_length_tmp * output_dims[-1]
        
        self.dropout = nn.Dropout(r_drop)
        self.linear = nn.Linear(final_dim, seq_length_out*n_feature_out)
        self.output_act = last_act

    def forward(self, x):
        ## x: (batch, seq, input_dim)

        batch_size = x.size(0)
        x = torch.transpose(x, 1, 2)
        ## x: (batch, input_dim, seq)

        for blk in self.blocks:
            x = blk(x)
        ## x: (batch, output_dims[-1], seq/2**n_layer)

        if "lstm" in self.additional_layer:
            x = torch.transpose(x, 1, 2)
            ## x: (batch, seq/2**n_layer, output_dims[-1])

            x, (h_t, c_t) = self.lstm(x)
            ## x: (batch, seq/2**n_layer, hidden_size)
            ## h_t: final hidden state (n_layer_lstm, batch, hidden_size)
            ## c_t: final cell state (n_layer_lstm, batch, hidden_size)
            ## h_t and c_t have batch at the second component even when batch_first = True
            
            x = x.contiguous().view(batch_size, -1)
            ## x: (batch, hidden_size * seq/2**n_layer)
            
            # Another possibility is to use the last hidden state
            # x = torch.transpose(h_t, 0, 1) ## x: (batch, n_layer_lstm, hidden_size)
            # x = x.contiguous().view(batch_size, -1) ## x: (batch, n_layer_lstm * hidden_size)

        elif self.additional_layer == "attention":
            x = torch.transpose(x, 1, 2)
            ## x: (batch, seq/2**n_layer, output_dims[-1])

            x, w = self.attention(x, x, x)
            ## x: (batch, seq/2**n_layer, output_dims[-1])
            ## w: weight (batch, seq/2**n_layer, output_dims[-1])

            x = x[:, 0, :]
            ## x: ( batch, output_dims[-1] )
        else:
            x = x.contiguous().view(batch_size, -1)
            ## x: (batch, hidden_dim*seq/2)

        x = self.dropout(x)
        x = self.linear(x)
        ## x: (batch, seq_length_out*n_feature_out)

        if self.seq_length_out != 1:
            x = x.reshape(batch_size, self.seq_length_out, -1)
            ## x: (batch, seq_length_out, n_feature_out) 

        x = self.output_act(x)

        return x

class Conv2dNet(nn.Module):

    def __init__(self, block, n_feature_in=8, n_feature_out=1, width=10, height=10, seq_length_out=10, hidden_dim=32, n_layer=4, kernel_size=3, r_drop=0, batch_norm=False, last_act=nn.LogSoftmax(dim=1)):
        super().__init__()

        self.seq_length_out = seq_length_out

        input_dims = [ n_feature_in ] + [ hidden_dim * min(2**i, 8) for i in range(n_layer-1) ]
        output_dims = [ hidden_dim * min(2**i, 8) for i in range(n_layer) ]

        if n_layer == 1:
            dropout_rates = [ r_drop ]
        else:
            dropout_rates = [0, 0] + [ r_drop for i in range(n_layer-2) ] 

        wtmp = width
        htmp = height
        kernel_sizes = []
        paddings = []
        ### e.g., for seq_length = 10 with 
        ### (input_dim, 10, 10) -> (hidden_dim*2, 5, 5) -> (hidden_dim*4, 3, 3) -> (hidden_dim*8, 2, 2) 
        for i in range(n_layer):
            wtmp = int( ( wtmp + 1 ) / 2 )
            htmp = int( ( htmp + 1 ) / 2 )
            if wtmp > kernel_size:
                if htmp > kernel_size:
                    kernel_sizes.append([kernel_size, kernel_size])
                else:
                    kernel_sizes.append([kernel_size, htmp])
            else:
                if htmp > kernel_size:
                    kernel_sizes.append([wtmp, kernel_size])
                else:
                    kernel_sizes.append([wtmp, htmp])
            paddings.append([ int( kw / 2 ) for kw in kernel_sizes[-1] ])

        self.blocks = nn.ModuleList([
            block(nin=i, nout=j, stride=2, kernel_size=ks, padding=pd, bn=batch_norm, r_drop=r)
            for i, j, ks, pd, r in zip(input_dims, output_dims, kernel_sizes, paddings, dropout_rates)
            ])

        final_dim = wtmp * htmp * output_dims[-1]
        self.linear = nn.Linear(final_dim, seq_length_out*n_feature_out)
        self.output_act = last_act

    def forward(self, x):
        ## x: (batch, input_dim, H, W)

        batch_size = x.size(0)

        for blk in self.blocks:
            x = blk(x)
        ## x: (batch, output_dims[-1], H/2**n_layer, W/2**n_layer)

        x = x.contiguous().view(batch_size, -1)
        ## x: (batch, hidden_dim*H*W/4)

        x = self.linear(x)
        ## x: (batch, seq_length_out*n_feature_out)

        if self.seq_length_out != 1:
            x = x.reshape(batch_size, self.seq_length_out, -1)
            ## x: (batch, seq_length_out, n_feature_out) 

        x = self.output_act(x)

        return x

class MixtureDensityNetwork(nn.Module):
    def __init__(self, block, K_mdn=5, n_feature_in=8, n_feature_out=1, seq_length=10, hidden_dim=32, n_layer=4, kernel_size=3, r_drop=0, batch_norm=False, last_act=nn.LogSoftmax(dim=1), nlayer_increase=3, additional_layer="" ):
        super().__init__()

        padding = int( kernel_size / 2 )

        input_dims = [ n_feature_in ] + [ hidden_dim * min(2**i, 2**(nlayer_increase-1)) for i in range(n_layer-1) ]
        output_dims = [ hidden_dim * min(2**i, 2**(nlayer_increase-1)) for i in range(n_layer) ]

        if n_layer == 1:
            dropout_rates = [ r_drop ]
        else:
            dropout_rates = [0, 0] + [ r_drop for i in range(n_layer-2) ] 

        self.blocks = nn.ModuleList([
            block(nin=i, nout=j, stride=2, kernel_size=kernel_size, padding=padding, bn=batch_norm, r_drop=r)
            for i, j, r in zip(input_dims, output_dims, dropout_rates)
            ])
        
        ### e.g., for seq_length = 10 with 
        ### (input_dim, 10) -> (hidden_dim*2, 5) -> (hidden_dim*4, 3) -> (hidden_dim*8, 2) 
        seq_length_tmp = seq_length
        for i in range(n_layer): seq_length_tmp = int( ( seq_length_tmp + 1 ) / 2 )

        self.additional_layer = additional_layer
        if "lstm" in self.additional_layer:
            self.hidden_size = 256
            self.n_layer_lstm = 2 if "2" in self.additional_layer else 1
            self.lstm = nn.LSTM(input_size=output_dims[-1], hidden_size=self.hidden_size, num_layers=self.n_layer_lstm, batch_first=True, dropout=r_drop)
            final_dim = seq_length_tmp * self.hidden_size
        elif self.additional_layer == "attention":
            self.attention = nn.MultiheadAttention(embed_dim=output_dims[-1], num_heads=8, batch_first=True)
            final_dim = seq_length_tmp * output_dims[-1] 
        else:
            final_dim = seq_length_tmp * output_dims[-1]
        
        self.dropout = nn.Dropout(r_drop)
        
        self.linear_pi = nn.Linear(final_dim, n_feature_out*K_mdn)
        self.linear_mu = nn.Linear(final_dim, n_feature_out*K_mdn)
        self.linear_sigma = nn.Linear(final_dim, n_feature_out*K_mdn)

        self.output_act_pi = nn.Softmax(dim=1)
        self.output_act_mu = last_act
        self.output_act_sigma = nn.Softplus()

    def forward(self, x):
        ## x: (batch, seq, input_dim)

        batch_size = x.size(0)
        x = torch.transpose(x, 1, 2)
        ## x: (batch, input_dim, seq)

        for blk in self.blocks:
            x = blk(x)
        ## x: (batch, output_dims[-1], seq/2**n_layer)

        if "lstm" in self.additional_layer:
            x = torch.transpose(x, 1, 2)
            ## x: (batch, seq/2**n_layer, output_dims[-1])

            x, (h_t, c_t) = self.lstm(x)
            ## x: (batch, seq/2**n_layer, hidden_size)
            ## h_t: final hidden state (n_layer_lstm, batch, hidden_size)
            ## c_t: final cell state (n_layer_lstm, batch, hidden_size)
            ## h_t and c_t have batch at the second component even when batch_first = True
            
            x = x.contiguous().view(batch_size, -1)
            ## x: (batch, hidden_size * seq/2**n_layer)
            
            # Another possibility is to use the last hidden state h_t
            # x = torch.transpose(h_t, 0, 1) ## x: (batch, n_layer_lstm, hidden_size)
            # x = x.contiguous().view(batch_size, -1) ## x: (batch, n_layer_lstm * hidden_size)

        elif self.additional_layer == "attention":
            x = torch.transpose(x, 1, 2)
            ## x: (batch, seq/2**n_layer, output_dims[-1])

            x, w = self.attention(x, x, x)
            ## x: (batch, seq/2**n_layer, output_dims[-1])
            ## w: weight (batch, seq/2**n_layer, output_dims[-1])

            x = x[:, 0, :]
            ## x: ( batch, output_dims[-1] )
        else:
            x = x.contiguous().view(batch_size, -1)
            ## x: (batch, hidden_dim*seq/2)

        x = self.dropout(x)
        x_pi = self.linear_pi(x)
        x_mu = self.linear_mu(x)
        x_sigma = self.linear_sigma(x)
        ## x: (batch, n_feature_out * K_mdn)

        x_pi = self.output_act_pi(x_pi)
        x_mu = self.output_act_mu(x_mu)
        x_sigma = self.output_act_sigma(x_sigma)

        return torch.cat([x_pi, x_mu, x_sigma], dim=1) # (batch, n_feature_out * K_mdn * 3)
