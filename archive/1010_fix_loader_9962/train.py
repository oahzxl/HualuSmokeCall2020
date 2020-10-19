import os
import time

import torch.utils.data
from tensorboardX import SummaryWriter
from torch import optim
from tqdm import tqdm

from eval import evaluate
from models import *
from utils.utils import sec2time


def train(model, train_loader, eval_loader, args):

    model.train()
    print("Start training")
    writer = SummaryWriter(log_dir=args.save_path)

    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothCELoss()
    # criterion = WeightedLabelSmoothCELoss(1978, 2168, 1227)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # warm_up_epochs = 5
    # warm_up_with_cosine_lr = lambda e: e / warm_up_epochs if e <= warm_up_epochs else 0.5 * (
    #         math.cos((e - warm_up_epochs) / (args.num_epochs - warm_up_epochs) * math.pi) + 1)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)  # CosineAnnealingLR

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    global_step, best_map, loss, t_remain = 0, 0, 0, 0

    for epoch in range(0, args.num_epochs, 1):
        running_loss = 0.0
        t = time.time()

        for i, data in enumerate(tqdm(train_loader)):
            image = data["image"].cuda()
            label = data["label"].cuda()

            # scheduler.step()
            optimizer.zero_grad()

            predict = model(image)
            loss = criterion(predict, label)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            if i % args.print_interval == 0 and i != 0:
                batch_time = (time.time() - t) / args.print_interval / args.batch_size
                running_loss = running_loss / args.print_interval
                print("==> [train] epoch = %2d, batch = %4d, global_step = %4d, loss = %.2f, "
                      "time per picture = %.2fs" % (epoch, i, global_step, running_loss, batch_time))
                writer.add_scalar("scalar/loss", running_loss, global_step, time.time())
                running_loss = 0.0
                t = time.time()
            global_step += 1

        print("==> [train] epoch = %2d, loss = %.2f, time per picture = %.2fs, remaining time = %s"
              % (epoch + 1, running_loss / len(train_loader),
                 (time.time() - t) / len(train_loader) / args.batch_size,
                 sec2time((time.time() - t_remain) * (args.num_epochs - epoch - 1)) if t_remain != 0 else '-1'))
        t_remain = time.time()
        print("==> [eval train] ", end='')
        map_on_train, acc_on_train, precision_on_train, recall_on_train, eval_loss = evaluate(
            train_loader, model, criterion)
        print("==> [eval valid] ", end='')
        map_on_valid, acc_on_valid, precision_on_valid, recall_on_valid, eval_loss = evaluate(
            eval_loader, model, criterion)

        # scheduler.step(eval_loss)  # ReduceLR

        writer.add_scalar("scalar/mAP_on_train", map_on_train, global_step, time.time())
        writer.add_scalar("scalar/mAP_on_train", map_on_valid, global_step, time.time())
        writer.add_scalar("scalar/accuracy_on_train", acc_on_train, global_step, time.time())
        writer.add_scalar("scalar/accuracy_on_valid", acc_on_valid, global_step, time.time())
        writer.add_scalar("scalar/precision_on_train", precision_on_train, global_step, time.time())
        writer.add_scalar("scalar/precision_on_valid", precision_on_valid, global_step, time.time())
        writer.add_scalar("scalar/recall_on_train", recall_on_train, global_step, time.time())
        writer.add_scalar("scalar/recall_on_valid", recall_on_valid, global_step, time.time())

        if map_on_valid > best_map:
            best_map = map_on_valid
            if float(map_on_valid) > 0.935:
                torch.save({
                    "model_state_dict": model.state_dict(),
                }, os.path.join(args.save_path, "%.5f" % best_map + ".tar"))
            print("==> [best] mAP: %.5f" % best_map)

    writer.close()
    print("Finish training.")
