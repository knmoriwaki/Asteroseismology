import os
import sys
import numpy as np

import re

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from tqdm import tqdm

def denorm(target, norm_params, n_feature_in, n_feature_out, output_dim, loss, batch=None):

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
    n_data = len(values)
    n_feature = len(values[0])

    weights = torch.zeros(n_data, n_feature)
    for j in range(n_feature):
        dx = ( xmax[j] - xmin[j] ) / len(pdf)

        weights_tot = 0.0
        for i, x in enumerate(values):
            ix = int( ( x - xmin[j] ) / dx )
            ix = np.clip(ix, 0, len(pdf)-1)
            weights[i][j] = 1. - pdf[ix][j]
            weights_tot += weights[i][j]

        for i in range(n_data):
            weights[i][j] /= weights_tot

    return weights

def print_pdf(pdf, xmin, xmax):
    dx = (xmax - xmin) / len(pdf)
    print("#### label distribution ####") 
    for i, p in enumerate(pdf):
        print("# {:e} {:e}".format(xmin+dx*(i+0.5), p))
    print("#### end of label distribution ####")


def load_fnames(data_dir, ndata, id_start=1, nrea_noise=1, r_train = 0.9, shuffle=True, suffix="data"):

    id_list = np.array(range(ndata))

    ### shuffle to randomly separate training and validation data
    if shuffle == True:
        np.random.shuffle(id_list)

    ### for training data, we use multiple realizations for each spectrum
    ### During the training, the data will be shuffled again via DataLoader
    ids_train = [ i for i in id_list[:int(ndata * r_train)]]
    fnames_train = [ "{}/{:07d}.{:d}.{}".format(data_dir, i+id_start, irea, suffix) for i in ids_train for irea in range(nrea_noise)]
    ids_train = [ i for i in ids_train for irea in range(nrea_noise) ]

    ### for validation data, we use only one realization (id=0) for each spectrum.
    ids_val = [ i for i in id_list[int(ndata * r_train):] ]
    fnames_val = [ "{}/{:07d}.0.{}".format(data_dir, i+id_start, suffix) for i in ids_val ]

    if len(ids_val) == 0:
        ids_val = [ids_train[-1]]
        fnames_val = [fnames_train[-1]]

    return fnames_train, fnames_val, ids_train, ids_val

def load_data(fnames, data_ids, fname_comb, output_dim, input_id, output_id, seq_length, norm_params=None, loss="l1norm", device="cpu", pbar=False):

    if len(np.shape(norm_params)) == 1:
        norm_params = norm_params.reshape(1,-1)

    print(f"Loading files... ", file=sys.stderr)

    ### read input data ###
    n_feature_in = len(input_id)
    data = []
    flist = tqdm(fnames, file=sys.stderr) if pbar else fnames
    count = 0
    for f in flist:
        if os.path.exists(f) == False: 
            #print(f"# Error: file not found {f}", file=sys.stderr) 
            #sys.exit(1)
            print(f"# Warning: file not found {f}", file=sys.stderr)
            print(f"# Delete data_ids {data_ids[count]}", file=sys.stderr)
            del data_ids[count]
            continue
        else:
            count += 1

        d = np.loadtxt(f)[:seq_length, input_id]
        data.append(d)

    data = np.array(data) #( ndata, seq_length)
    data = data.reshape( -1, seq_length, 1 ) #( ndata, seq_length, n_feature_in)

    if np.shape(data)[1] != seq_length:
        print(f"Error: inconsistent seq_length {np.shape(data)[1]} != {seq_length}", file=sys.stderr)
        sys.exit(1)

    if np.shape(data)[2] != n_feature_in:
        print(f"Error: inconsistent n_feature_in {np.shape(data)[2]} != {n_feature_in}", file=sys.stderr)
        sys.exit(1)


    ### read target data ###
    n_feature_out = len(output_id)
    target = np.loadtxt(fname_comb, skiprows=0, usecols=output_id, ndmin=2)
    target = target[data_ids] # (ndata, n_feature_out)

    fname_id = np.loadtxt(fname_comb, skiprows=0, usecols=0)
    fname_id = fname_id[data_ids]
    fname_id0 = "%07d" % int(fname_id[0]) 

    if fname_id0 not in fnames[0]:
        print("Error: id %d is inconsistent with fname %s" % (fname_id[0], fnames[0]), file=sys.stderr)
        print("Check skiprows and id_start in utils.py", file=sys.stderr)
        sys.exit(1)

    # if you want to convert it to sin, do so here by, e.g., 
    #target[:,0] = np.sin( np.deg2rad( target[:,0] ))
    # in this case, do not forget to change the normalization parameter accordingly

    if norm_params is not None:
        ## normalize input data
        for i in range(n_feature_in):
            print(i)
            data[:,:,i] -= norm_params[i,0]
            if norm_params[i,1] > 0: data[:,:,i] /= norm_params[i,1]
            print("# input feature {:d}: xmin = {:.1f}, dx = {:.1f}".format(i, norm_params[i,0], norm_params[i,1]))
        ## normalize target data
        for ii in range(n_feature_out):
            i = n_feature_in + ii
            target[:,ii] -= norm_params[i,0]
            if norm_params[i,1] > 0: target[:,ii] /= norm_params[i,1]
            print("# output feature {:d}: xmin = {:.1f}, dx = {:.1f}".format(ii, norm_params[i,0], norm_params[i,1]))

    if loss == "nllloss":
        target = np.array([ [ int( t * output_dim ) if t < 1 else output_dim -1 for t in tt ] for tt in target ])
        # (ndata, n_feature_out)

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


def load_data_2d(fnames, data_ids, fname_comb, output_dim, input_id, output_id, width, height, norm_params=None, loss="l1norm", device="cpu", pbar=False):

    if len(np.shape(norm_params)) == 1:
        norm_params = norm_params.reshape(1,-1)

    print(f"Loading files... ", file=sys.stderr)

    ### read input data ###
    n_feature_in = len(input_id)
    data = []
    flist = tqdm(fnames, file=sys.stderr) if pbar else fnames
    count = 0
    for f in flist:
        if os.path.exists(f) == False: 
            #print(f"# Error: file not found {f}", file=sys.stderr) 
            #sys.exit(1)
            print(f"# Warning: file not found {f}", file=sys.stderr)
            print(f"# Delete data_ids {data_ids[count]}", file=sys.stderr)
            del data_ids[count]
            continue
        else:
            count += 1

        d = np.loadtxt(f)
        data.append(d)

    data = np.array(data)  #( ndata, width, height)
    data = data.reshape( -1, 1, width, height ) #( ndata, n_feature_in, width, height)

    if np.shape(data)[1] != n_feature_in:
        print(f"Error: inconsistent n_feature_in {np.shape(data)[1]} != {n_feature_in}", file=sys.stderr)
        sys.exit(1)

    if np.shape(data)[2] != width:
        print(f"Error: inconsistent seq_length {np.shape(data)[2]} != {width}", file=sys.stderr)
        sys.exit(1)

    if np.shape(data)[3] != height:
        print(f"Error: inconsistent seq_length {np.shape(data)[3]} != {height}", file=sys.stderr)
        sys.exit(1)


    ### read target data ###
    n_feature_out = len(output_id)
    target = np.loadtxt(fname_comb, skiprows=5, usecols=output_id, ndmin=2)
    target = target[data_ids] # (ndata, n_feature_out)

    fname_id = np.loadtxt(fname_comb, skiprows=5, usecols=0)
    fname_id = fname_id[data_ids]
    fname_id0 = "%07d" % int(fname_id[0]) 

    if fname_id0 not in fnames[0]:
        print("Error: id %d is inconsistent with fname %s" % (fname_id[0], fnames[0]), file=sys.stderr)
        print("Check skiprows and id_start in utils.py", file=sys.stderr)
        sys.exit(1)

    # if you want to convert it to sin, do so here by, e.g., 
    #target[:,0] = np.sin( np.deg2rad( target[:,0] ))
    # in this case, do not forget to change the normalization parameter accordingly

    if norm_params is not None:
        ## normalize input data
        for i in range(n_feature_in):
            data[:,i] -= norm_params[i,0]
            if norm_params[i,1] > 0: data[:,i] /= norm_params[i,1]
            print("# input feature {:d}: xmin = {:.1f}, dx = {:.1f}".format(i, norm_params[i,0], norm_params[i,1]))
        ## normalize target data
        for ii in range(n_feature_out):
            i = n_feature_in + ii
            target[:,ii] -= norm_params[i,0]
            if norm_params[i,1] > 0: target[:,ii] /= norm_params[i,1]
            print("# output feature {:d}: xmin = {:.1f}, dx = {:.1f}".format(ii, norm_params[i,0], norm_params[i,1]))

    if loss == "nllloss":
        target = np.array([ [ int( t * output_dim ) if t < 1 else output_dim -1 for t in tt ] for tt in target ])
        # (ndata, n_feature_out)

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
            


