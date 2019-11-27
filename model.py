import numpy as np

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNetNoFC(nn.Module):

    def __init__(self, to_select=[3], num_classes=1000):
        super(AlexNetNoFC, self).__init__()
        self._to_select = to_select
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        desired_feats = dict() # key is layer number, value is FloatTensor
        layer = 0
        for module in list(self.features.children()):
            x = module(x)
            if layer in self._to_select:
                desired_feats[layer] = x
            layer += 1

        return desired_feats

def alexnet(pretrained=True, **kwargs):
    model1 = AlexNetNoFC(**kwargs)
    state_dict = model_zoo.load_url(model_urls['alexnet'])

    #print 'Model1: automatic weight set'
    #print model1._modules

    if pretrained:
        model1.load_state_dict(state_dict)

    return model1

if __name__ == "__main__":
    import torch
    from torch.autograd import Variable

    idx = 7
    img_size = 64
    a = alexnet(pretrained=True, to_select=[idx])
    opt_img = Variable(torch.rand(1,3,img_size,img_size))

    act = a(opt_img)
    print act[idx].size()

    from torchvision.models import resnet50
    m = resnet50(pretrained=True)

