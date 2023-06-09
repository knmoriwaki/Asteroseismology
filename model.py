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
    elif args.model == "CNN":
        model = ConvNet(Conv1dBlock, n_feature_in=n_feature_in, n_feature_out=n_feature_out, seq_length=args.seq_length, seq_length_out=args.output_dim, hidden_dim=args.hidden_dim, n_layer=args.n_layer, r_drop=args.r_drop, last_act=last_act)
    elif args.model == "CNN_LSTM":
        model = ConvNet(Conv1dBlock, n_feature_in=n_feature_in, n_feature_out=n_feature_out, seq_length=args.seq_length, seq_length_out=args.output_dim, hidden_dim=args.hidden_dim, n_layer=args.n_layer, r_drop=args.r_drop, last_act=last_act, additional_layer="lstm")
    elif args.model == "CNN_at":
        model = ConvNet(Conv1dBlock, n_feature_in=n_feature_in, n_feature_out=n_feature_out, seq_length=args.seq_length, seq_length_out=args.output_dim, hidden_dim=args.hidden_dim, n_layer=args.n_layer, r_drop=args.r_drop, last_act=last_act, additional_layer="attention")
    elif args.model == "CNN2":
        model = ConvNet(Conv1dBlock2, n_feature_in=n_feature_in, n_feature_out=n_feature_out, seq_length=args.seq_length, seq_length_out=args.output_dim, hidden_dim=args.hidden_dim, n_layer=args.n_layer, r_drop=args.r_drop, last_act=last_act)
    elif args.model == "ResNet":
        model = ResNet(ResidualBlock, [3,3,3,3], n_feature_in=n_feature_in, n_feature_out=n_feature_out, seq_length=args.seq_length, output_dim=args.output_dim, hidden_dim=args.hidden_dim, last_act=last_act)
    elif args.model == "BNN":
        if args.loss == "nllloss":
            print("Error: The current version does not allow nllloss for BNN", file=sys.stderr)
            sys.exit(1)
        model = ConvNet(Conv1dBlock_w_Conv2d, n_feature_in=n_feature_in, n_feature_out=n_feature_out, seq_length=args.seq_length, seq_length_out=args.output_dim, hidden_dim=args.hidden_dim, n_layer=args.n_layer, r_drop=args.r_drop, last_act=last_act)
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

class Conv1dBlock2(nn.Module):
    def __init__(self, nin=32, nout=32, kernel_size=3, stride=1, padding="same", bn=False, r_drop=0):
        super().__init__()

        self.bn = bn
        self.drop = True if r_drop > 0 else False

        self.conv = nn.Conv1d(nin, nout, kernel_size=kernel_size, stride=1, padding="same")
        if self.bn:
            self.batch_norm = nn.BatchNorm1d(nout)
        if self.drop:
            self.drop = nn.Dropout(r_drop)
        self.act = nn.LeakyReLU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=stride, padding=1)


    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.batch_norm(x)
        if self.drop:
            x = self.drop(x)
        x = self.act(x)
        x = self.pool(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super().__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm1d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

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

    def __init__(self, block, n_feature_in=8, n_feature_out=1, seq_length=10, seq_length_out=10, hidden_dim=32, n_layer=4, kernel_size=3, r_drop=0, last_act=nn.LogSoftmax(dim=1), additional_layer=None):
        super().__init__()

        self.seq_length_out = seq_length_out

        padding = int( kernel_size / 2 )

        input_dims = [ n_feature_in ] + [ hidden_dim * min(2**i, 8) for i in range(n_layer-1) ]
        output_dims = [ hidden_dim * min(2**i, 8) for i in range(n_layer) ]
        #batch_norms = [ True ] + [ False ] * (n_layer-1)
        batch_norms = [ False ] * n_layer
        if n_layer == 1:
            dropout_rates = [ r_drop ]
        else:
            dropout_rates = [0, 0] + [ r_drop for i in range(n_layer-2) ] 

        self.blocks = nn.ModuleList([
            block(nin=i, nout=j, stride=2, kernel_size=kernel_size, padding=padding, bn=bn, r_drop=r)
            for i, j, bn, r in zip(input_dims, output_dims, batch_norms, dropout_rates)
            ])

        self.additional_layer = additional_layer
        if self.additional_layer == "lstm":
            self.hidden_size = 1024
            self.n_layer_lstm = 1
            self.lstm = nn.LSTM(input_size=output_dims[-1], hidden_size=self.hidden_size, num_layers=self.n_layer_lstm, batch_first=True)
            self.linear = nn.Linear(self.hidden_size, seq_length_out*n_feature_out)
        elif self.additional_layer == "attention":
            self.attention = nn.MultiheadAttention(embed_dim=output_dims[-1], num_heads=8, batch_first=True)
            self.linear = nn.Linear(output_dims[-1], seq_length_out*n_feature_out)
            
        else:
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
        ## x: (batch, output_dims[-1], seq/2**n_layer)

        if self.additional_layer == "lstm":
            x = torch.transpose(x, 1, 2)
            ## x: (batch, seq/2**n_layer, output_dims[-1])

            x, (h_t, c_t) = self.lstm(x)
            ## x: (batch, seq/2**n_layer, hidden_size)
            ## h_t: final hidden state (n_layer_lstm, batch, hidden_size)
            ## c_t: final cell state (n_layer_lstm, batch, hidden_size)
            ## h_t and c_t have batch at the second component even when batch_first = True
            
            x = h_t[-1,:,:]
            ## x: (batch, hidden_size)
        elif self.additional_layer == "attention":
            x = torch.transpose(x, 1, 2)
            ## x: (batch, seq/2**n_layer, output_dims[-1])

            x = self.attention(x, x, x)
            ## x: (batch, seq/2**n_layer, output_dims[-1])

            x = x[:, 0, :]
            ## x: ( batch, output_dims[-1] )
        else:
            x = x.contiguous().view(batch_size, -1)
            ## x: (batch, hidden_dim*seq/2)

        x = self.linear(x)
        ## x: (batch, seq_length_out*n_feature_out)

        if self.seq_length_out != 1:
            x = x.reshape(batch_size, self.seq_length_out, -1)
            ## x: (batch, seq_length_out, n_feature_out) 

        x = self.output_act(x)

        return x


class ResNet(nn.Module):
    #https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
    def __init__(self, block, layers, n_feature_in, n_feature_out, seq_length, output_dim=1, hidden_dim=64, last_act=nn.Identity()):
        super().__init__()
        self.inplanes = hidden_dim
        self.output_dim = output_dim
        self.conv1 = nn.Sequential(
                        nn.Conv1d(n_feature_in, hidden_dim, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU())
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, hidden_dim, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, hidden_dim*2, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, hidden_dim*4, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, hidden_dim*8, layers[3], stride = 2)
        self.avgpool = nn.AvgPool1d(7, stride=1)
        self.fc = nn.Linear(hidden_dim*8*1414, n_feature_out*output_dim)
        self.last_act = last_act
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = torch.transpose(x, 1, 2) #x: (batch, input_dim, seq)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.output_dim > 1:
            x = x.view(x.size(0), self.output_dim, -1)
        x = self.last_act(x)

        return x
