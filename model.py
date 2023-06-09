import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

def MyModel(args):
    if args.model == "RNN":
        model = RecurrentNet(input_dim=args.n_feature, hidden_dim=args.hidden_dim, output_dim=args.output_dim, n_layer=args.n_layer)
    elif args.model == "CNN":
        model = ConvNet(input_dim=args.n_feature, seq_length=args.seq_length, hidden_dim=args.hidden_dim, output_dim=args.output_dim, n_layer=args.n_layer, r_drop=args.r_drop)
    elif args.model == "BNN":
        model = ConvNet(input_dim=args.n_feature, seq_length=args.seq_length, hidden_dim=args.hidden_dim, output_dim=args.output_dim, n_layer=args.n_layer)
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

    def __init__(self, nin=32, nout=32, kernel_size=5, stride=2, padding="same", r_drop=0):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=(kernel_size,1), stride=(stride,1), padding=(padding,0))
        self.drop = nn.Dropout(r_drop)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = x.unsqueeze(3)
        x = self.conv(x)
        x = self.drop(x)
        x = self.act(x)
        x = x.squeeze(3)
        return x

"""
class Conv1dBlock(nn.Module):

    def __init__(self, nin=32, nout=32, kernel_size=5, stride=2, padding="same", r_drop=0):
        super().__init__()
        self.conv = nn.Conv1d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding)
        self.drop = nn.Dropout(r_drop)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.drop(x)
        x = self.act(x)
        return x
"""

class RecurrentNet(nn.Module):

    def __init__(self, input_dim=8, hidden_dim=32, output_dim=1, n_layer=1, nonlinearity="tanh", r_drop=0, last_act=nn.LogSoftmax(dim=1)):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer

        self.rnn = nn.RNN(input_dim, hidden_dim, n_layer, nonlinearity=nonlinearity, batch_first=True, dropout=r_drop, bidirectional=False) 
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.output_act = last_act

    def forward(self, x):
        ## x: (batch, seq, input_dim)

        out, h_last = self.rnn(x)
        ## out: (batch, seq, hidden_dim)
        ## h_last: (n_layer, batch, hidden_dim)

        out = out[:,-1,:].contiguous().view(-1, self.hidden_dim)
        ## out: (batch, hidden_dim)

        out = self.linear1(out)
        out = self.linear2(out)
        out = self.output_act(out)
        ## out: (batch, output_dim)

        return out

class ConvNet(nn.Module):

    def __init__(self, input_dim=8, seq_length=10, hidden_dim=32, output_dim=1, n_layer=4, kernel_size=5, r_drop=0, last_act=nn.LogSoftmax(dim=1)):
        super().__init__()

        padding = int( kernel_size / 2 )

        input_dims = [ input_dim ] + [ hidden_dim * 2**i for i in range(n_layer-1) ]
        output_dims = [ hidden_dim * 2**i for i in range(n_layer) ]
        if n_layer == 1:
            dropout_rates = [ r_drop ]
        else:
            dropout_rates = [0] + [ r_drop for i in range(n_layer-1) ] 
        self.blocks = nn.ModuleList([
            Conv1dBlock(nin=i, nout=j, stride=2, kernel_size=kernel_size, padding=padding, r_drop=r)
            for i, j, r in zip(input_dims, output_dims, dropout_rates) 
            ])
        ### for seq_length = 10 with 
        ### (input_dim, 10) -> (hidden_dim*2, 5) -> (hidden_dim*4, 3) -> (hidden_dim*8, 2) 
        tmp = seq_length
        for i in range(n_layer): tmp = int( ( tmp + 1 ) / 2 )
        final_dim = tmp * output_dims[-1]
        
        self.linear = nn.Linear(final_dim, output_dim)
        self.output_act = last_act

    def forward(self, x):
        ## x: (batch, seq, input_dim)

        batch_size = x.size(0)
        x = torch.transpose(x, 1, 2)
        ## x: (batch, input_dim, seq)

        for blk in self.blocks:
            x = blk(x)
        ## x: (batch, hidden_dim*2**(n_layer-1), seq/2**n_layer)

        x = x.contiguous().view(batch_size, -1)
        ## x: (batch, hidden_dim*seq/2)

        x = self.linear(x)
        x = self.output_act(x)

        return x



