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
parser.add_argument("--isTrain", dest="isTrain", action='store_true', help="train or test")
parser.add_argument("--data_dir", dest="data_dir", default="./Data_analysis", help="Root directory of training dataset")
parser.add_argument("--test_dir", dest="test_dir", default="./test_data", help="Root directory of test data")
parser.add_argument("--ndata", dest="ndata", type=int, default=10, help="the number of data")
parser.add_argument("--nrea_noise", dest="nrea_noise", type=int, default=1, help="the number of data")
parser.add_argument("--model_dir", dest="model_dir", default="./Model", help="Root directory to save learned model parameters")
parser.add_argument("--output_id", dest="output_id", nargs="+", type=int, default=13, help="the column number in Combinations.txt. You can put multiple ids.")

parser.add_argument("--model", dest="model", default="NN", help="model")
parser.add_argument("--fname_norm", dest="fname_norm", default="./norm_params.txt", help="file name of the normalization parameters")
parser.add_argument("--n_feature", dest="n_feature", type=int, default=1, help="number of input elements")
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
args = parser.parse_args()

def main():

    is_cuda = torch.cuda.is_available()
    random_seed = 1
    if is_cuda:
        device = torch.device("cuda:0")

        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True

        print("# GPU is available")
    else:
        device = torch.device("cpu")
        print("# GPU not availabe, CPU used")

    np.random.seed(random_seed)

    if args.isTrain:
        with open("{}/params.json".format(args.model_dir), mode="a") as f:
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
            loss_func = nn.L1Loss(reduction=reduction)
        elif "bce" in args.loss:
            loss_func = nn.BCELoss(reduction=reduction)
        else:
            print("Error: unknown loss", file=sys.stderr)
            sys.exit(1)

    print( f"# loss function: {args.loss}")


    ### define network and optimizer ###
    model = MyModel(args) 
    print(model)
    summary( model, input_size=(args.batch_size, args.seq_length, args.n_feature), col_names=["output_size", "num_params"])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0) #default: lr=1e-3, betas=(0.9,0.999), eps=1e-8
    def lambda_rule(ee):
        lr_l = 1.0 - max(0, ee + 1 - args.epoch) / float( args.epoch_decay + 1 )
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    model.to(device)

    print( f"# hidden_dim: {args.hidden_dim}" )
    print( f"# n_layer: {args.n_layer}" )

    ### load training and validation data ###
    norm_params = np.loadtxt(args.fname_norm)
    train_fnames, val_fnames, train_ids, val_ids, = load_fnames(args.data_dir, ndata=args.ndata, nrea_noise=args.nrea_noise, r_train=0.9, shuffle=True)
    fname_comb = f"{args.data_dir}/Combinations.txt"

    data, label = load_data(train_fnames, train_ids, fname_comb, output_dim=args.output_dim, output_id=args.output_id, n_feature=args.n_feature, seq_length=args.seq_length, norm_params=norm_params, loss=args.loss, device=None)
    val_data, val_label = load_data(val_fnames, val_ids, fname_comb, output_dim=args.output_dim, output_id=args.output_id, n_feature=args.n_feature, seq_length=args.seq_length, norm_params=norm_params, loss=args.loss, device=device)

    dataset = MyDataset(data, label)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    ntrain = label.size(dim = 0)

    if "weighted" in args.loss:
        nbin = 20
        pdf = torch.histc(label, bins=nbin, min=0, max=0)
        pdf = pdf / ntrain
        pdf.to(device)
        hist_min = label.min()
        hist_max = label.max()
        print_pdf(pdf, hist_min, hist_max)

    ### training ###
    idx = 0
    n_per_epoch = float( int( ntrain / args.batch_size ) )
    print("Training...", file=sys.stderr)
    fout = "{}/log.txt".format(args.model_dir)
    print(f"# output {fout}", file=sys.stderr)
    with open(fout, "w") as f:
        print("#idx loss loss_val", file=f)
    for ee in tqdm(range(args.epoch + args.epoch_decay), file=sys.stderr):
        #for ee in range(args.epoch + args.epoch_decay):
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
        fname = "{}/val.txt".format(args.model_dir)
        with open(fname, "w") as f:
            for i, (ll, oo) in enumerate(zip(val_label, output)):
                if args.loss == "nllloss":
                    oo = torch.argmax(oo, dim=0)

                pred = denormalization(oo, norm_params, args.n_feature, n_feature_out, args.output_dim, args.loss)
                true = denormalization(ll, norm_params, args.n_feature, n_feature_out, args.output_dim, args.loss)

                for j in range(n_feature_out):
                    print(true[0,j].item(), pred[0,j].item(), end=" ",  file=f)
                print("", file=f)

        print(f"# output {fname}", file=sys.stderr)

        if args.loss == "nllloss":
            for i, oo in enumerate(output):
                fname = "{}/val_dist{:d}.txt".format(args.model_dir, i)
                with open(fname, "w") as f:
                    for iclass in range(args.output_dim):
                        print((iclass+0.5)/args.output_dim, end=" ", file=f)
                        for j in range(n_feature_out):
                            print(oo[iclass,j].item(), end=" ", file=f)
                        print("", file=f)
                print(f"# output {fname}", file=sys.stderr)

        if args.model == "BNN":
            #model.unfreeze()

            output_list = []

            true = denormalization(val_label, norm_params, args.n_feature, n_feature_out, args.output_dim, args.loss)
            for i in range(100):
                output = model(val_data)
                if args.loss == "nllloss":
                    output = torch.argmax(output, dim=1)
                output = denormalization(output, norm_params, args.n_feature, n_feature_out, args.output_dim, args.loss)
                output_list.append(output)
            pred = torch.mean(output_list, axis=0)
            pred_std = torch.std(output_list, axis=0)

            fname = "{}/val_bnn.txt".format(args.model_dir)
            with open(fname, "w") as f:
                for i in range(len(val_data)):
                    for j in range(n_feature_out):
                        print(true[i,j].item(), pred[i,j].item(), pred_std[i,j].item(), file=f)
            print(f"# output {fname}", file=sys.stderr)


    ### save model ###
    fsave = "{}/model.pth".format(args.model_dir)
    torch.save(model.state_dict(), fsave)
    print( f"# save {fsave}" )



####################################################
### test
####################################################
def test(device):

    n_feature_out = 1 if isinstance(args.output_id, int) else len(args.output_id)

    ### define network ###
    model = MyModel(args)
    model.to(device)

    fmodel = "{}/model.pth".format(args.model_dir)
    model.load_state_dict(torch.load(fmodel))
    model.eval()
    print("# load model from {}/model.pth".format(args.model_dir))

    ### load test data ###
    norm_params = np.loadtxt(args.fname_norm)
    _, test_fnames, _, test_ids = load_fnames(args.test_dir, args.ndata, r_train=0.0, shuffle=False)
    fname_comb = f"{args.test_dir}/Combinations.txt"
    data, label = load_data(test_fnames, test_ids, fname_comb, args.output_dim, output_id=args.output_id, n_feature=args.n_feature, seq_length=args.seq_length, norm_params=norm_params, loss=args.loss, device=None)

    ### output test result ###
    fname = "{}/test.txt".format(args.model_dir)
    with open(fname, "w") as f:
        for dd, ll in zip(data, label):

            dd = torch.unsqueeze(dd, dim=0).to(device)
            ll = torch.unsqueeze(ll, dim=0).to(device)

            output = model(dd)
            if args.loss == "nllloss":
                output = torch.argmax(output, dim=1)
            pred = denormalization(output, norm_params, args.n_feature, n_feature_out, args.output_dim, args.loss)
            true = denormalization(ll, norm_params, args.n_feature, n_feature_out, args.output_dim, args.loss)
            for j in range(n_feature_out):
                print(true[0,j].item(), pred[0,j].item(), end=" ", file=f)
            print("", file=f)

            del dd, ll, output
            torch.cuda.empty_cache()
    print(f"output {fname}", file=sys.stderr)

if __name__ == "__main__":
    main()
