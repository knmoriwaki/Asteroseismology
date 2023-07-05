import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchhk import transform_model
import torchbnn as bnn

def MyModel(args):

    n_feature_out = 1 if isinstance(args.output_id, int) else len(args.output_id)

    if args.loss == "nllloss":
        last_act = nn.LogSoftmax(dim=1)
    else:
        args.output_dim = 1
        last_act = nn.Sigmoid() 

    if args.model == "RNN":
        model = RecurrentNet(n_feature_in=args.n_feature, n_feature_out=n_feature_out, seq_length_out=args.output_dim, hidden_dim=args.hidden_dim, n_layer=args.n_layer, last_act=last_act)
    elif args.model == "CNN":
        model = ConvNet(n_feature_in=args.n_feature, n_feature_out=n_feature_out, seq_length=args.seq_length, seq_length_out=args.output_dim, hidden_dim=args.hidden_dim, n_layer=args.n_layer, r_drop=args.r_drop, last_act=last_act)
    elif args.model == "BNN":
        if args.loss == "nllloss":
            print("Error: The current version does not allow nllloss for BNN", file=sys.stderr)
            sys.exit(1)
        model = ConvNet(n_feature_in=args.n_feature, n_feature_out=n_feature_out, seq_length=args.seq_length, seq_length_out=args.output_dim, hidden_dim=args.hidden_dim, n_layer=args.n_layer, r_drop=args.r_drop, last_act=last_act)
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

    def __init__(self, nin=32, nout=32, kernel_size=5, stride=2, padding="same", r_drop=0, bn=False):
        super().__init__()

        self.bn = bn

        self.conv = nn.Conv1d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm1d(nout)
        self.drop = nn.Dropout(r_drop)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.batch_norm(x)
        x = self.drop(x)
        x = self.act(x)
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

    def __init__(self, n_feature_in=8, n_feature_out=1, seq_length=10, seq_length_out=10, hidden_dim=32, n_layer=4, kernel_size=5, r_drop=0, last_act=nn.LogSoftmax(dim=1)):
        super().__init__()

        self.seq_length_out = seq_length_out

        padding = int( kernel_size / 2 )

        input_dims = [ n_feature_in ] + [ hidden_dim * min(2**i, 32) for i in range(n_layer-1) ]
        output_dims = [ hidden_dim * min(2**i, 32) for i in range(n_layer) ]
        batch_norms = [ True for i in range(n_layer-1) ]
        if n_layer == 1:
            dropout_rates = [ r_drop ]
        elif n_layer == 2:
            dropout_rates = [ 0, r_drop ]
        else:
            dropout_rates = [0] + [ r_drop for i in range(n_layer-1) ] 

        self.blocks = nn.ModuleList([
            Conv1dBlock(nin=i, nout=j, stride=2, kernel_size=kernel_size, padding=padding, r_drop=r, bn=bn)
            for i, j, r, bn in zip(input_dims, output_dims, dropout_rates, batch_norms) 
            ])

        ### e.g., for seq_length = 10 with 
        ### (input_dim, 10) -> (hidden_dim*2, 5) -> (hidden_dim*4, 3) -> (hidden_dim*8, 2) 
        tmp = seq_length
        for i in range(n_layer): tmp = int( ( tmp + 1 ) / 2 )
        final_dim = tmp * output_dims[-1]
        
        self.linear = nn.Linear(final_dim, seq_length_out*n_feature_out)
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
        ## x: (batch, seq_length_out*n_feature_out)

        if self.seq_length_out != 1:
            x = x.reshape(batch_size, self.seq_length_out, -1)
            ## x: (batch, seq_length_out, n_feature_out) 

        x = self.output_act(x)

        return x



