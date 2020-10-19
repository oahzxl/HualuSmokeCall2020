import argparse
import os

import torch.utils.data

from models import *
from test import test
from train import train
from utils import data_loader


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train", help='train/test')
    # parser.add_argument('--mode', type=str, default="test", help='train/test')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--model', type=str, default="resnet18")
    # parser.add_argument('--model', type=str, default="resnext101")
    # parser.add_argument('--model', type=str, default="resnext_wsl")

    parser.add_argument('--num-epochs', type=float, default=20)
    parser.add_argument('--print-interval', type=int, default=9999)
    parser.add_argument('--num-attentions', type=int, default=8)
    parser.add_argument('--input-size', type=tuple, default=(409, 538))
    parser.add_argument('--beta', type=float, default=5e-2)
    parser.add_argument('--num-workers', type=int, default=16)

    parser.add_argument('--data-path', type=str, default="./data/")
    parser.add_argument('--ckp-path', type=str, default="./log/train.tar")
    parser.add_argument('--save-path', type=str, default="./log/tmp/")
    if not os.path.exists(parser.parse_args().save_path):
        os.mkdir(parser.parse_args().save_path)

    return parser.parse_args()


if __name__ == "__main__":

    args = init_args()
    print(args)
    print("Start loading data")

    if args.model == "wsdan":
        model = WSDAN(num_classes=2, M=32, net='inception_mixed_6e', pretrained=False)
    elif args.model == "resnext101":
        model = resnext101_32x8d(pretrained=True, progress=True)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 3)
        )
    elif args.model == "resnext_wsl":
        model = resnext101_32x8d_wsl(progress=True)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 3)
        )
    elif args.model == "resnet18":
        model = resnet18(pretrained=True, progress=True)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 3)
        )
    else:
        raise ValueError
    model.cuda()

    if args.mode == "train":
        train_dataset = data_loader.build_dataset_train(args.data_path, args.input_size)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, shuffle=True, drop_last=True)
        eval_dataset = data_loader.build_dataset_eval(args.data_path, args.input_size)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers, shuffle=True, drop_last=True)
        train(model, train_loader, eval_loader, args)

    if args.mode == "test":
        test_dataset = data_loader.build_dataset_test(args.data_path, args.input_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        test(test_loader, model, args.ckp_path)
