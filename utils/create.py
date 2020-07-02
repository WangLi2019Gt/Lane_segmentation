from net import deeplab
import torch.nn.functional as F
from utils.computation import *


def create_net(in_channels, num_classes, net_name,backbone=None):

    if net_name == 'deeplab':
        if backbone:
            network = deeplab.DeepLab(backbone=backbone,num_classes=num_classes)
        else:
            network = deeplab.DeepLab(num_classes=num_classes)
    else:
        raise ValueError('Not supported Net_name: {}'.format(net_name))
    return network


def create_loss(predicts, labels, num_classes):

    predicts = predicts.permute((0, 2, 3, 1)).reshape((-1,num_classes))
    bce_loss = F.cross_entropy(predicts, labels.flatten(), reduction='mean')
    one_hot_label=onehot_encoder(labels.reshape((-1,1)),num_classes).to(labels.device)
    loss = bce_loss+DiceLoss()(predicts,one_hot_label)
    miou = compute_miou(predicts,labels.reshape((-1,1)),num_classes)
    return loss, miou

