import torch
import torch.utils.data
from torch.nn.functional import softmax
from utils.utils import batch_augment
from utils.voc2010 import VOCMApMetric


def evaluate(data_loader, model, criterion, args):

    with torch.no_grad():

        model.eval()
        meter = VOCMApMetric(class_names=("normal", "smoke", "call"))
        total_loss, correct, total, tp, fp, tn, fn = 0, 0, 0, 0, 0, 0, 0

        for data in data_loader:
            image = data["image"].cuda()
            label = data["label"].cuda()

            if args.model == "wsdan":
                y_pred_raw, feature_matrix, attention_map = model(image)
                with torch.no_grad():
                    crop_images = batch_augment(image, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6),
                                                padding_ratio=0.1)
                y_pred_crop, _, _ = model(crop_images)
                loss = (criterion(y_pred_raw, label) + criterion(y_pred_crop, label)) / 2
                predict = (y_pred_raw + y_pred_crop) / 2

            else:
                predict = model(image)
                loss = criterion(predict, label)

            predict = softmax(predict, dim=-1)
            pred_labels = torch.argmax(predict, dim=-1)
            pred_bboxes = torch.ones((pred_labels.size(0), 1, 4))
            gt_bboxes = torch.ones((pred_labels.size(0), 1, 4))
            meter.update(pred_bboxes.cpu().numpy(), pred_labels.cpu().numpy(), predict.cpu().numpy(),
                         gt_bboxes.cpu().numpy(), label.cpu().numpy())

            _, predict = predict.max(1)
            total_loss += loss
            total += label.size(0)
            correct += predict.eq(label).sum().item()
            tp += torch.sum(predict & label)
            fp += torch.sum(predict & (1 - label))
            tn += torch.sum((1 - predict) & (1 - label))
            fn += torch.sum((1 - predict) & label)

        loss = total_loss / len(data_loader)
        acc = float(correct) / float(total)
        precision = (float(tp) + 1e-6) / (float(tp + fp) + 1e-3)
        recall = (float(tp) + 1e-6) / (float(tp + fn) + 1e-3)
        m_ap_normal, m_ap_smoke, m_ap_calling, m_ap = meter.get()[1]
        print("mAP = %.3f, mAP_n = %.3f, mAP_s = %.3f, mAP_c = %.3f, "
              "loss = %.3f, acc = %.3f, p = %.3f, r = %.3f"
              "" % (m_ap, m_ap_normal, m_ap_smoke, m_ap_calling, loss, acc, precision, recall))

    return m_ap, acc, precision, recall, loss
