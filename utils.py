import os
import sys
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from tqdm import tqdm


def calc_weight(pdf, values, xmin, xmax):
    dx = ( xmax - xmin ) / len(pdf)

    weights = torch.zeros(len(values))
    for i, x in enumerate(values):
        ix = int( ( x - xmin ) / dx )
        ix = np.clip(ix, 0, len(pdf)-1)
        weights[i] = 1. - pdf[ix]
    return torch.reshape(weights, (-1,1))

def print_pdf(pdf, xmin, xmax):
    dx = (xmax - xmin) / len(pdf)
    print("#### label distribution ####") 
    for i, p in enumerate(pdf):
        print("# {:e} {:e}".format(xmin+dx*(i+0.5), p))
    print("#### end of label distribution ####")


def load_fnames(data_dir, ndata, r_train = 0.9, shuffle=True):

    id_list = np.array(range(ndata))
    if shuffle == True:
        np.random.shuffle(id_list)

    ids_train = [ i for i in id_list[:int(ndata * r_train)] ]
    ids_val = [ i for i in id_list[int(ndata * r_train):] ]

    fnames_train = [ "{}/{:07d}.0.data".format(data_dir, i) for i in ids_train ]
    fnames_val = [ "{}/{:07d}.0.data".format(data_dir, i) for i in ids_val ]

    if len(ids_val) == 0:
        ids_val = [ids_train[-1]]
        fnames_val = [fnames_train[-1]]

    return fnames_train, fnames_val, ids_train, ids_val

def load_data(fnames, data_ids, fname_comb="./Combinations.txt", output_dim=100, output_id=[13], n_feature=1, seq_length=10, norm_params=None, loss="l1norm", data_aug=[], device="cpu"):

    if len(np.shape(norm_params)) == 1:
        norm_params = norm_params.reshape(1,-1)

    print(f"Reading files... (data_aug = {data_aug})", file=sys.stderr)

    ### read input data ###
    data = []
    for f in fnames:
        if os.path.exists(f) == False: 
            print(f"# Warning: file not found {f}", file=sys.stderr) 
            continue 

        for da in [0] + data_aug:
            if da == 0:
                d = read_data( f, norm_params=norm_params)
            else: 
                print("data augmentation {} is not defined")
                sys.exit(1)

            data.append(d)
    
    ### read label data ###
    label = np.loadtxt(fname_comb, skiprows=5, usecols=output_id)
    label = label[data_ids] # (ndata)
    xmin = 0.0
    xmax = 90.001
    if loss == "nllloss":
        dx = ( xmax - xmin ) / output_dim
        label = [ int( ( l - xmin ) / dx ) for l in label ]  ### this is for nllloss and doesn't work properly for other losses.
        if np.max( label ) >= output_dim:
            print(f"Error: label value {np.max(label)} is greater than output_dim {output_dim}")
            sys.exit(1)
    else:
        label = ( label - xmin ) / ( xmax - xmin )
        if len(np.shape(label)) == 1:
            label = label.reshape(-1, 1) #(ndata, 1) within [0,1]
        if np.shape(label)[1] != output_dim:
            print(f"Error: inconsistent output_dim {np.shape(label)[1]} != {output_dim}", file=sys.stderr)
            sys.exit(1)

    if np.shape(data)[1] != seq_length:
        print(f"Error: inconsistent seq_length {np.shape(data)[1]} != {seq_length}", file=sys.stderr)
        sys.exit(1)

    if np.shape(data)[2] != n_feature:
        print(f"Error: inconsistent n_feature {np.shape(data)[2]} != {n_feature}", file=sys.stderr)
        sys.exit(1)

    ### convert the data to torch.tensor ###
    data = torch.from_numpy( np.array(data).astype(np.float32) )
    label = torch.from_numpy( np.array(label) )
    if loss == "nllloss":
        label = label.to(torch.long)
    else:
        label = label.to(torch.float32)

    ### send the data to device ###
    if device is not None:
        data = data.to(device)
        label = label.to(device)

    print( f"# data size: {data.size()}" )
    print( f"# label size: {label.size()}" )

    return data, label

def read_data(path, norm_params=None):

    ## source data ## 
    data = np.loadtxt(path)
    input_data = data[:,1] ## read "spec"
    if len(np.shape(input_data)) == 1:
        input_data = input_data.reshape(-1,1) #(seq_length, n_feature=1)

    if norm_params is not None: 
        for i in range(np.shape(norm_params)[0]):
            input_data[:,i] -= norm_params[i,0]
            if norm_params[i,1] > 0: input_data[:,i] /= norm_params[i,1]

    return input_data

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data, label):
        self.transform = None #transforms.Compose([transforms.ToTensor()])
        self.data = data
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
            out_data = self.transform(self.data)[0][idx]
            out_label = self.label[idx]
        else:
            out_data = self.data[idx]
            out_label = self.label[idx]
        return out_data, out_label
            


