import json

import torch
import torch.utils.data
from torch.nn.functional import softmax
from tqdm import tqdm

from utils.utils import batch_augment
from utils.utils import load_model


def test(data_loader, model, ckp_path, args):

    with torch.no_grad():

        optimizer = torch.optim.Adagrad(model.parameters())
        model = load_model(model, optimizer, ckp_path)
        model.eval()

        outputs = []
        idx_to_name = ['normal', 'smoking', 'calling']

        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):

            image = data["image"].cuda()
            name = data["name"]

            if args.model == "wsdan":
                y_pred_raw, _, attention_map = model(image)
                with torch.no_grad():
                    crop_images = batch_augment(image, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6),
                                                padding_ratio=0.1)
                y_pred_crop, _, _ = model(crop_images)
                output = (y_pred_raw + y_pred_crop) / 2

            else:
                output = model(image)
            output = softmax(output, dim=-1)

            for j in range(image.size(0)):

                idx = torch.argmax(output[j, :])
                category = idx_to_name[idx]
                score = output[j, idx]

                outputs.append({
                    "category": category,
                    "image_name": name[j],
                    "score": round(float(score), 5)
                })

    outputs.sort(key=lambda x: int(x['image_name'].split('.')[0]))
    with open("./log/result.json", "w+") as f:
        json.dump(outputs, f, indent=4)

    print("Done.")
    return 0
