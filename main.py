import sys
import argparse
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from torchinfo import summary

from model import MyModel, RecurrentNet, ConvNet

from tqdm import tqdm

from utils import *

parser = argparse.ArgumentParser(description="")
parser.add_argument("--gpu_id", dest="gpu_id", type=int, default=0, help="")
parser.add_argument("--isTrain", dest="isTrain", action='store_true', help="train or test")
parser.add_argument("--data_dir", dest="data_dir", default="./Data_analysis", help="Root directory of training dataset")
parser.add_argument("--test_dir", dest="test_dir", default="./test_data", help="Root directory of test data")
parser.add_argument("--ndata", dest="ndata", type=int, default=10, help="the number of data")
parser.add_argument("--r_train", dest="r_train", type=float, default=0.9, help="ratio of number of training data to ndata")
parser.add_argument("--nrea_noise", dest="nrea_noise", type=int, default=1, help="the number of data")
parser.add_argument("--model_dir_save", dest="model_dir_save", default="./Model", help="Root directory to save learned model parameters")
parser.add_argument("--model_dir_load", dest="model_dir_load", default="./Model", help="Root directory to load learned model parameters")
parser.add_argument("--input_id", dest="input_id", nargs="+", type=int, default=1, help="the column number(s) in, e.g., 0000000.0.txt.")
parser.add_argument("--output_id", dest="output_id", nargs="+", type=int, default=13, help="the column number(s) in Combinations.txt. You can put multiple ids.")

parser.add_argument("--model", dest="model", default="NN", help="model")
parser.add_argument("--fname_norm", dest="fname_norm", default="./norm_params.txt", help="file name of the normalization parameters")

parser.add_argument("--seq_length", dest="seq_length", type=int, default=45412, help="length of the sequence to input into RNN")
parser.add_argument("--hidden_dim", dest="hidden_dim", type=int, default=32, help="number of NN nodes")
parser.add_argument("--output_dim", dest="output_dim", type=int, default=30, help="the output dimension for nllloss. Not used for the other loss functions")
parser.add_argument("--n_layer", dest="n_layer", type=int, default=5, help="number of NN layers")
parser.add_argument("--r_drop", dest="r_drop", type=float, default=0.0, help="dropout rate")
parser.add_argument("--batch_size", dest="batch_size", type=int, default=4, help="batch size")
parser.add_argument("--epoch", dest="epoch", type=int, default=10, help="training epoch")
parser.add_argument("--epoch_decay", dest="epoch_decay", type=int, default=0, help="training epoch")
parser.add_argument("--lr", dest="lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--loss", dest="loss", default="l1norm", help="loss function")

parser.add_argument("--i_layer_freeze", dest="i_layer_freeze", nargs="+", type=int, default=-1, help="layer numbers (0, 1, ..., n_layer-1) to be freezed. You can freeze multple layers.")

parser.add_argument("--progress_bar", action="store_true")


args = parser.parse_args()

def main():

    random_seed = 1

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device_name = "cuda:{:d}".format(args.gpu_id)
        device = torch.device(device_name)

        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True

        print("# GPU ({}) is available".format(device_name))
    else:
        device = torch.device("cpu")
        print("# GPU not availabe, CPU used")

    np.random.seed(random_seed)

    if args.isTrain:
        with open("{}/params.json".format(args.model_dir_save), mode="a") as f:
            json.dump(args.__dict__, f)
        train(device)
    else:
        test(device)


def update_learning_rate(optimizer, scheduler):
    old_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('# learning rate %.7f -> %.7f' % (old_lr, lr))

####################################################
### training
####################################################

def train(device):

    n_feature_in = 1 if isinstance(args.input_id, int) else len(args.input_id)
    n_feature_out = 1 if isinstance(args.output_id, int) else len(args.output_id)

    ### define loss function ###
    if args.loss == "nllloss":
        loss_func = nn.NLLLoss(reduction="mean")
        if args.output_dim < 2:
            print("Error: output_dim should be greater than 1 for NLLLoss", file=sys.stderr)
            sys.exit(1)
    else:

        reduction = "mean"
        if "weighted" in args.loss:
            reduction = "none"

        if "l1norm" in args.loss:
            loss_func = nn.SmoothL1Loss(reduction=reduction)
        elif "bce" in args.loss:
            loss_func = nn.BCELoss(reduction=reduction)
        else:
            print("Error: unknown loss", file=sys.stderr)
            sys.exit(1)

    print( f"# loss function: {args.loss}")


    ### define network ###
    model = MyModel(args) 

    print(model)
    #summary( model, input_size=(args.batch_size, args.seq_length, n_feature_in), col_names=["output_size", "num_params"], device=device)

    if args.model_dir_load != args.model_dir_save and args.model_dir_load != "./Model": ### load network parameters and freeze the first few layers of the network

        fmodel = "{}/model.pth".format(args.model_dir_load)
        model.load_state_dict(torch.load(fmodel))
        print("# load model from {}".format(fmodel))

        if isinstance(args.i_layer_freeze, int):
            args.i_layer_freeze = [args.i_layer_freeze]
        for child in model.children():
            for i, c in enumerate(child.children()):
                if i in args.i_layer_freeze:
                    print("# Freezee parameters of module below")
                    print(c)
                    for param in c.parameters():
                        param.requires_grad = False

    model.to(device)

    ### define optimizer ###

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0) #default: lr=1e-3, betas=(0.9,0.999), eps=1e-8
    def lambda_rule(ee):
        lr_l = 1.0 - max(0, ee + 1 - args.epoch) / float( args.epoch_decay + 1 )
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


    print( f"# hidden_dim: {args.hidden_dim}" )
    print( f"# n_layer: {args.n_layer}" )

    ### load training and validation data ###
    norm_params = np.loadtxt(args.fname_norm)
    train_fnames, val_fnames, train_ids, val_ids, = load_fnames(args.data_dir, ndata=args.ndata, nrea_noise=args.nrea_noise, r_train=args.r_train, shuffle=True)
    fname_comb = f"{args.data_dir}/../Combinations.txt"

    data, label = load_data(train_fnames, train_ids, fname_comb, output_dim=args.output_dim, input_id=args.input_id, output_id=args.output_id, seq_length=args.seq_length, norm_params=norm_params, loss=args.loss, device=None, pbar=args.progress_bar)
    val_data, val_label = load_data(val_fnames, val_ids, fname_comb, output_dim=args.output_dim, input_id=args.input_id, output_id=args.output_id, seq_length=args.seq_length, norm_params=norm_params, loss=args.loss, device=device, pbar=args.progress_bar)

    dataset = MyDataset(data, label)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    ntrain = label.size(dim = 0)

    if "weighted" in args.loss:
        nbin = 20
        pdf = np.zeros((n_feature_out, nbin))
        hist_min = np.zeros(n_feature_out)
        hist_max = np.zeros(n_feature_out)
        for i in range(n_feature_out):
            pdf[i] = torch.histc(label[:,i], bins=nbin, min=0, max=0)
            pdf[i] = pdf[i] / ntrain
            hist_min[i] = label.min()
            hist_max[i] = label.max()
            print_pdf(pdf[i], hist_min[i], hist_max[i])
        
        pdf = torch.from_numpy(pdf).to(device)

    ### training ###
    idx = 0
    n_per_epoch = float( int( ntrain / args.batch_size ) )
    print("Training...", file=sys.stderr)
    fout = "{}/log.txt".format(args.model_dir_save)
    print(f"# output {fout}", file=sys.stderr)
    with open(fout, "w") as f:
        print("#idx loss loss_val", file=f)
    
    elist = range(args.epoch + args.epoch_decay)
    if args.progress_bar:
        elist = tqdm(elist, file=sys.stderr)
    for ee in elist:
        if ee != 0:
            update_learning_rate(optimizer, scheduler)
        for i, (dd, ll) in enumerate(train_loader):
            
            dd = dd.to(device)
            ll = ll.to(device)
            output = model(dd)

            model.eval()
            with torch.no_grad():
                output_val = model(val_data)
            model.train()

            if "weighted" in args.loss:
                weights = calc_weight(pdf, output, hist_min, hist_max).to(device)
                weights_val = calc_weight(pdf, output_val, hist_min, hist_max).to(device) 
                loss = torch.mean( nbin * weights / weights.sum() * loss_func(output, ll) )
                loss_val = torch.mean( nbin * weights_val / weights_val.sum() * loss_func(output_val, val_label) )
                del weights, weights_val
            else:
                loss = loss_func(output, ll)
                loss_val = loss_func(output_val, val_label)

            print("{:d} {:f} {:f} {:f}".format(idx, idx/n_per_epoch, loss.item(), loss_val.item()) )
            with open(fout, "a") as f:
                print("{:d} {:f} {:f} {:f}".format(idx, idx/n_per_epoch, loss.item(), loss_val.item()), file=f)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            del dd, ll, output
            torch.cuda.empty_cache()

            idx += 1
 
    ### print validation result ###
    with torch.no_grad():
        output = model(val_data)
        fname = "{}/val.txt".format(args.model_dir_save)
        with open(fname, "w") as f:
            for i, (ll, oo) in enumerate(zip(val_label, output)):
                if args.loss == "nllloss":
                    oo = torch.argmax(oo, dim=0)

                pred = denorm(oo, norm_params, n_feature_in, n_feature_out, args.output_dim, args.loss)
                true = denorm(ll, norm_params, n_feature_in, n_feature_out, args.output_dim, args.loss)

                for j in range(n_feature_out):
                    print(true[0,j].item(), pred[0,j].item(), end=" ",  file=f)
                print(val_ids[i], file=f)

        print(f"# output {fname}", file=sys.stderr)

        if args.loss == "nllloss":
            for i, oo in enumerate(output):
                fname = "{}/val_dist{:d}.txt".format(args.model_dir_save, i)
                with open(fname, "w") as f:
                    for iclass in range(args.output_dim):
                        print((iclass+0.5)/args.output_dim, end=" ", file=f)
                        for j in range(n_feature_out):
                            print(oo[iclass,j].item(), end=" ", file=f)
                        print("", file=f)
                print(f"# output {fname}", file=sys.stderr)

        if args.model == "BNN":
            #model.unfreeze()


            true = denorm(val_label, norm_params, n_feature_in, n_feature_out, args.output_dim, args.loss)
            nsample = 100
            output_list = torch.zeros(nsample, len(val_data), n_feature_out)
            for i in range(nsample):
                output = model(val_data)
                if args.loss == "nllloss":
                    output = torch.argmax(output, dim=1)
                output = denorm(output, norm_params, n_feature_in, n_feature_out, args.output_dim, args.loss)
                output_list[i] = output
            pred = torch.mean(output_list, axis=0)
            pred_std = torch.std(output_list, axis=0)

            fname = "{}/val_bnn.txt".format(args.model_dir_save)
            with open(fname, "w") as f:
                for i in range(len(val_data)):
                    for j in range(n_feature_out):
                        print(true[i,j].item(), pred[i,j].item(), pred_std[i,j].item(), file=f)
            print(f"# output {fname}", file=sys.stderr)


    ### save model ###
    fsave = "{}/model.pth".format(args.model_dir_save)
    torch.save(model.state_dict(), fsave)
    print( f"# save {fsave}" )



####################################################
### test
####################################################
def test(device):

    n_feature_in = 1 if isinstance(args.input_id, int) else len(args.input_id)
    n_feature_out = 1 if isinstance(args.output_id, int) else len(args.output_id)

    ### define network ###
    model = MyModel(args)
    model.to(device)

    fmodel = "{}/model.pth".format(args.model_dir_load)
    model.load_state_dict(torch.load(fmodel))
    model.eval()
    print("# load model from {}".format(fmodel))

    ### load test data ###
    norm_params = np.loadtxt(args.fname_norm)
    _, test_fnames, _, test_ids = load_fnames(args.test_dir, args.ndata, r_train=0.0, shuffle=False)
    fname_comb = f"{args.test_dir}/../Combinations.txt"
    data, label = load_data(test_fnames, test_ids, fname_comb, args.output_dim, input_id=args.input_id, output_id=args.output_id, seq_length=args.seq_length, norm_params=norm_params, loss=args.loss, device=None, pbar=args.progress_bar)

    ### output test result ###
    fname = "{}/test.txt".format(args.model_dir_save)
    with open(fname, "w") as f:
        for i, (dd, ll) in enumerate(zip(data, label)):

            dd = torch.unsqueeze(dd, dim=0).to(device)
            ll = torch.unsqueeze(ll, dim=0).to(device)

            output = model(dd)
            if args.loss == "nllloss":
                output = torch.argmax(output, dim=1)

            pred = denorm(output, norm_params, n_feature_in, n_feature_out, args.output_dim, args.loss)
            true = denorm(ll, norm_params, n_feature_in, n_feature_out, args.output_dim, args.loss)

            for j in range(n_feature_out):
                print(true[0,j].item(), pred[0,j].item(), end=" ", file=f)
            print(test_ids[i], file=f)

            del dd, ll, output
            torch.cuda.empty_cache()

    print(f"output {fname}", file=sys.stderr)

if __name__ == "__main__":
    main()
