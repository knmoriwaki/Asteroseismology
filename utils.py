import os
import sys
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from tqdm import tqdm

def denormalization(target, norm_params, n_feature_in, n_feature_out, output_dim, loss, batch=None):

    ## target should have a shape (batch, n_feature_out)
    target = target.reshape(-1, n_feature_out)

    ### for "nllloss", convert the integers in [0,output_dim-1] to "floats" in [0,1]
    if loss == "nllloss":
        target = ( target + 0.5 ) / output_dim

    ### Then convert the label in [0,1] to the values.
    if norm_params is not None:
        for ii in range(n_feature_out):
            i = n_feature_in + ii
            if norm_params[i,1] > 0: target[:,ii] *= norm_params[i,1]
            target[:,ii] += norm_params[i,0]


    return target

def calc_weight(pdf, values, xmin, xmax):
    dx = ( xmax - xmin ) / len(pdf)

    weights = torch.zeros(len(values))
    for i, x in enumerate(values):
        ix = int( ( x - xmin ) / dx )
        ix = np.clip(ix, 0, len(pdf)-1)
        weights[i] = 1. - pdf[ix]
        weights_tot += weights[i]

    return torch.reshape(weights/weights_tot, (-1,1))

def print_pdf(pdf, xmin, xmax):
    dx = (xmax - xmin) / len(pdf)
    print("#### label distribution ####") 
    for i, p in enumerate(pdf):
        print("# {:e} {:e}".format(xmin+dx*(i+0.5), p))
    print("#### end of label distribution ####")


def load_fnames(data_dir, ndata, id_start=1, nrea_noise=1, nrea_noise_val=3, r_train = 0.9, shuffle=True):

    id_list = np.array(range(ndata))

    ### shuffle to randomly separate training and validation data
    if shuffle == True:
        np.random.shuffle(id_list)

    ### for training data, we use multiple realizations for each spectrum
    ### During the training, the data will be shuffled again via DataLoader
    ids_train = [ i for i in id_list[:int(ndata * r_train)]]
    fnames_train = [ "{}/{:07d}.{:d}.data".format(data_dir, i+id_start, irea) for i in ids_train for irea in range(nrea_noise)]
    ids_train = [ i for i in ids_train for irea in range(nrea_noise) ]

    ### for validation data, we use only one realization (id=0) for each spectrum.
    ids_val = [ i for i in id_list[int(ndata * r_train):] ]
    fnames_val = [ "{}/{:07d}.0.data".format(data_dir, i+id_start) for i in ids_val ]

    if len(ids_val) == 0:
        ids_val = [ids_train[-1]]
        fnames_val = [fnames_train[-1]]

    return fnames_train, fnames_val, ids_train, ids_val

def load_data(fnames, data_ids, fname_comb="./Combinations.txt", output_dim=100, output_id=[13], n_feature=1, seq_length=10, norm_params=None, loss="l1norm", device="cpu"):

    if len(np.shape(norm_params)) == 1:
        norm_params = norm_params.reshape(1,-1)

    print(f"Reading files... ", file=sys.stderr)

    ### read input data ###
    data = []
    for f in fnames:
        if os.path.exists(f) == False: 
            print(f"# Error: file not found {f}", file=sys.stderr) 
            sys.exit(1)

        input_data = np.loadtxt(f)
        d = input_data[:,1] ## read "spec"
        d = d.reshape(seq_length, n_feature) #(seq_length, n_feature=1)
            
        data.append(d) #( ndata, seq_length, n_feature)
    data = np.array(data)

    ### read target data ###
    n_feature_out = len(output_id)
    target = np.loadtxt(fname_comb, skiprows=5, usecols=output_id)
    target = target[data_ids] # (ndata, n_feature_out)
    if n_feature_out == 1:
        target = target.reshape(-1,1) # (ndata, n_feature_out=1)

    # if you want to convert it to sin, do so here by, e.g., 
    # target[:,0] = np.sin( np.deg2rad( target[:,0] ))
    # in this case, do not forget to change the normalization parameter accordingly

    if norm_params is not None:
        ## normalize input data
        for i in range(n_feature):
            data[:,:,i] -= norm_params[i,0]
            if norm_params[i,1] > 0: data[:,:,i] /= norm_params[i,1]
        ## normalize target data
        for ii in range(n_feature_out):
            i = n_feature + ii
            target[:,ii] -= norm_params[i,0]
            if norm_params[i,1] > 0: target[:,ii] /= norm_params[i,1]

    if loss == "nllloss":
        target = np.array([ [ int( t * output_dim ) if t < 1 else output_dim -1 for t in tt ] for tt in target ])
        # (ndata, n_feature_out)

    if np.shape(data)[1] != seq_length:
        print(f"Error: inconsistent seq_length {np.shape(data)[1]} != {seq_length}", file=sys.stderr)
        sys.exit(1)

    if np.shape(data)[2] != n_feature:
        print(f"Error: inconsistent n_feature {np.shape(data)[2]} != {n_feature}", file=sys.stderr)
        sys.exit(1)

    ### convert the data to torch.tensor ###
    data = torch.from_numpy( np.array(data).astype(np.float32) )
    target = torch.from_numpy( np.array(target) )
    if loss == "nllloss":
        target = target.to(torch.long)
    else:
        target = target.to(torch.float32)

    ### send the data to device ###
    if device is not None:
        data = data.to(device)
        target = target.to(device)

    print( f"# data size: {data.size()}" )
    print( f"# target size: {target.size()}" )

    return data, target

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
            


