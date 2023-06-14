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
parser.add_argument("--model_dir", dest="model_dir", default="./Model", help="Root directory to save learned model parameters")
parser.add_argument("--output_id", dest="output_id", nargs="+", type=int, default=13, help="the column number in Combinations.txt. You can put multiple ids.")

parser.add_argument("--model", dest="model", default="NN", help="model")
parser.add_argument("--fname_norm", dest="fname_norm", default="./norm_params.txt", help="file name of the normalization parameters")
parser.add_argument("--n_feature", dest="n_feature", type=int, default=1, help="number of input elements")
parser.add_argument("--seq_length", dest="seq_length", type=int, default=45412, help="length of the sequence to input into RNN")
parser.add_argument("--hidden_dim", dest="hidden_dim", type=int, default=32, help="number of NN nodes")
parser.add_argument("--output_dim", dest="output_dim", type=int, default=1, help="the output dimension")
parser.add_argument("--n_layer", dest="n_layer", type=int, default=5, help="number of NN layers")
parser.add_argument("--r_drop", dest="r_drop", type=float, default=0.0, help="dropout rate")
parser.add_argument("--batch_size", dest="batch_size", type=int, default=4, help="batch size")
parser.add_argument("--epoch", dest="epoch", type=int, default=10, help="training epoch")
parser.add_argument("--epoch_decay", dest="epoch_decay", type=int, default=0, help="training epoch")
parser.add_argument("--lr", dest="lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--loss", dest="loss", default="l1norm", help="loss function")
args = parser.parse_args()

xmin = 0.0
xmax = 90.001
dx = ( xmax - xmin ) / args.output_dim

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

    ### define network, optimizer, and loss function ###
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

    if args.loss == "nllloss":
        loss_func = nn.NLLLoss(reduction="mean")
    elif args.loss == "l1norm":
        loss_func = nn.L1Loss(reduction="mean")
    else:
        print("Error: unknown loss", file=sys.stderr)
        sys.exit(1)

    print( f"# loss function: {args.loss}")

    ### load training and validation data ###
    norm_params = np.loadtxt(args.fname_norm)
    train_fnames, val_fnames, train_ids, val_ids, = load_fnames(args.data_dir, ndata=args.ndata, r_train=0.9, shuffle=True)
    fname_comb = f"{args.data_dir}/Combinations.txt"

    data, label = load_data(train_fnames, train_ids, fname_comb, output_dim=args.output_dim, output_id=args.output_id, n_feature=args.n_feature, seq_length=args.seq_length, norm_params=norm_params, loss=args.loss, device=None)
    val_data, val_label = load_data(val_fnames, val_ids, fname_comb, output_dim=args.output_dim, output_id=args.output_id, n_feature=args.n_feature, seq_length=args.seq_length, norm_params=norm_params, loss=args.loss, device=device)

    dataset = MyDataset(data, label)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    ntrain = label.size(dim = 0)

    ### training ###
    idx = 0
    print("Training...", file=sys.stderr)
    fout = "{}/log.txt".format(args.model_dir)
    with open(fout, "w") as f:
        print("#idx loss loss_val", file=f)
    for ee in tqdm(range(args.epoch + args.epoch_decay), file=sys.stderr):
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
                weights = w1 * ll + w0 * ( 1. - ll )
                weights_val = w1 * val_label + w0 * ( 1. - val_label )
                loss = torch.mean( nbin * weights / weights.sum() * loss_func(output, ll) )
                loss_val = torch.mean( nbin * weights_val / weights_val.sum() * loss_func(output_val, val_label) )
            else:
                loss = loss_func(output, ll)
                loss_val = loss_func(output_val, val_label)

            print("{:d} {:f} {:f}".format(idx, loss.item(), loss_val.item()) )
            with open(fout, "a") as f:
                print("{:d} {:f} {:f}".format(idx, loss.item(), loss_val.item()), file=f)

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
                    id_max = torch.argmax(oo)
                    pred = xmin + dx * (id_max + 0.5)
                    true = xmin + dx * (ll + 0.5)
                else:
                    pred = oo * ( xmax - xmin ) + xmin
                    true = ll * ( xmax - xmin ) + xmin
                print(true.item(), pred.item(), file=f)
        print(f"# output {fname}", file=sys.stderr)

        if args.loss == "nllloss":
            for i, oo in enumerate(output):
                fname = "{}/val_dist{:d}.txt".format(args.model_dir, i)
                with open(fname, "w") as f:
                    for iclass in range(args.output_dim):
                        print(xmin + dx*(iclass+0.5), oo[iclass].item(), file=f)
                print(f"# output {fname}", file=sys.stderr)

    ### save model ###
    fsave = "{}/model.pth".format(args.model_dir)
    torch.save(model.state_dict(), fsave)
    print( f"# save {fsave}" )



####################################################
### test
####################################################
def test(device):

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
                id_max = torch.argmax(output)
                pred = xmin + dx * (id_max + 0.5)
                true = xmin + dx * (ll + 0.5)
            else:
                pred = output * ( xmax - xmin ) + xmin
                true = ll * ( xmax - xmin ) + xmin
            print(true.item(), pred.item(), file=f)

            del dd, ll, output
            torch.cuda.empty_cache()
    print(f"output {fname}", file=sys.stderr)

if __name__ == "__main__":
    main()
