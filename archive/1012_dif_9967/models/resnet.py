import torchvision
import torch

def model_init(device_ids, num_classes=2):
    model = torchvision.models.resnet101(num_classes=num_classes)
    #model = torch.nn.DataParallel(model, device_ids=device_ids) # 声明所有可用设备
    #model = model.cuda(device=device_ids[0])  # 模型放在主设备
    model.cuda()
    return model