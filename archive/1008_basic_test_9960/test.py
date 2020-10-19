import json

import torch
import torch.utils.data
from torch import optim
from torch.nn.functional import softmax
from tqdm import tqdm

from utils.utils import load_model


def test(data_loader, model, ckp_path):

    with torch.no_grad():

        optimizer = torch.optim.Adagrad(model.parameters())
        model, _, _, _, _ = load_model(model, optimizer, ckp_path)
        model.eval()

        outputs = []
        idx_to_name = ['normal', 'smoking', 'calling']

        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):

            image = data["image"].cuda()
            name = data["name"]

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
