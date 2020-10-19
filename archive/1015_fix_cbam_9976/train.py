import os
import time

import torch.utils.data
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from tqdm import tqdm

from eval import evaluate
from utils.utils import batch_augment, sec2time


def train(model, train_loader, eval_loader, args):

    model.train()
    print("Start training")
    writer = SummaryWriter(log_dir=args.save_path)

    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothCELoss()
    # criterion = WeightedLabelSmoothCELoss(1978, 2168, 1227)

    fc_params = list(map(id, model.fc.parameters()))
    # fc_params += list(map(id, model.ca.parameters()))
    # fc_params += list(map(id, model.sa.parameters()))
    base_params = filter(lambda p: id(p) not in fc_params, model.parameters())
    if args.optim == 'Adam':
        optimizer = optim.Adam([{'params': base_params, 'lr': args.lr / 10},
                               {'params': model.fc.parameters()},
                                # {'params': model.ca.parameters()},
                                # {'params': model.sa.parameters()}
                                ], lr=args.lr)
    elif args.optim == 'SGD':
        optimizer = optim.SGD([{'params': base_params, 'lr': args.lr / 10},
                               {'params': model.fc.parameters()}], lr=args.lr, momentum=0.9)
    else:
        raise ValueError

    if args.sche == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=args.factor, patience=args.patience)
    elif args.sche == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=args.tm)
    elif args.sche == 'None':
        scheduler = None
    else:
        raise ValueError

    global_step, best_map, loss, t_remain, best_loss = 0, 0, 0, 0, 999.0

    for epoch in range(0, args.num_epochs, 1):
        running_loss = 0.0
        t = time.time()

        for i, data in enumerate(tqdm(train_loader)):
            image = data["image"].cuda()
            label = data["label"].cuda()

            optimizer.zero_grad()

            if args.model == "wsdan":
                y_pred_raw, feature_matrix, attention_map = model(image)
                with torch.no_grad():
                    crop_images = batch_augment(image, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6),
                                                padding_ratio=0.1)
                y_pred_crop, _, _ = model(crop_images)
                loss = (criterion(y_pred_raw, label) + criterion(y_pred_crop, label)) / 2
            else:
                predict = model(image)
                loss = criterion(predict, label)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            if args.sche == 'cos':
                scheduler.step(epoch + i / len(train_loader))

            if i % args.print_interval == 0 and i != 0:
                batch_time = (time.time() - t) / args.print_interval / args.batch_size
                running_loss = running_loss / args.print_interval
                print("==> [train] epoch = %2d, batch = %4d, global_step = %4d, loss = %.2f, "
                      "time per picture = %.2fs" % (epoch, i, global_step, running_loss, batch_time))
                writer.add_scalar("scalar/loss", running_loss, global_step, time.time())
                running_loss = 0.0
                t = time.time()
            global_step += 1

        print("[train] epoch = %2d, loss = %.4f, lr = %.1e, time per picture = %.2fs, remaining time = %s"
              % (epoch + 1, running_loss / len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'],
                 (time.time() - t) / len(train_loader) / args.batch_size,
                 sec2time((time.time() - t_remain) * (args.num_epochs - epoch - 1)) if t_remain != 0 else '-1'))
        t_remain = time.time()

        print("[eval train] ", end='')
        map_on_train, acc_on_train, precision_on_train, recall_on_train, loss_on_train = evaluate(
            train_loader, model, criterion, args)
        print("[eval valid] ", end='')
        map_on_valid, acc_on_valid, precision_on_valid, recall_on_valid, loss_on_valid = evaluate(
            eval_loader, model, criterion, args)

        if args.sche == 'reduce':
            # scheduler.step(loss_on_valid)  # ReduceLR
            scheduler.step(map_on_valid)  # ReduceLR

        writer.add_scalar("scalar/loss_on_train", loss_on_train, global_step, time.time())
        writer.add_scalar("scalar/loss_on_valid", loss_on_valid, global_step, time.time())
        writer.add_scalar("scalar/mAP_on_train", map_on_train, global_step, time.time())
        writer.add_scalar("scalar/mAP_on_valid", map_on_valid, global_step, time.time())
        writer.add_scalar("scalar/accuracy_on_train", acc_on_train, global_step, time.time())
        writer.add_scalar("scalar/accuracy_on_valid", acc_on_valid, global_step, time.time())
        writer.add_scalar("scalar/precision_on_train", precision_on_train, global_step, time.time())
        writer.add_scalar("scalar/precision_on_valid", precision_on_valid, global_step, time.time())
        writer.add_scalar("scalar/recall_on_train", recall_on_train, global_step, time.time())
        writer.add_scalar("scalar/recall_on_valid", recall_on_valid, global_step, time.time())

        if loss_on_valid < best_loss:
            best_loss = loss_on_valid
            print("==> [best] loss: %.5f" % best_loss)
        if float(loss_on_valid) < 0.16:
            torch.save({
                "model_state_dict": model.state_dict(),
            }, os.path.join(args.save_path, args.model + "_loss_%.5f" % best_loss + ".tar"))

        if map_on_valid > best_map:
            best_map = map_on_valid
            print("==> [best] mAP: %.5f" % best_map)
        if float(map_on_valid) > 0.93:
            torch.save({
                "model_state_dict": model.state_dict(),
            }, os.path.join(args.save_path, args.model + "_mAP_%.5f" % best_map + ".tar"))

    writer.close()
    print("Done.")
