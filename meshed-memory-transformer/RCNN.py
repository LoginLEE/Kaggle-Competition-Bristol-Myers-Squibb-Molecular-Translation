import torch
import torchvision
import torch.nn as nn
RCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None)
RCNN.eval()
#print(RCNN.roi_heads)# = nn.Identity

features = {}
def get_activation(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

RCNN.roi_heads.box_head.register_forward_hook(get_activation('feature'))
#RCNN.roi_heads.box_predictor.cls_score = nn.Identity()
#print('-'*50)
out = RCNN(torch.rand(1,3,224,224))#
#print(RCNN.roi_heads)
#print(out)

print('act',features['feature'])