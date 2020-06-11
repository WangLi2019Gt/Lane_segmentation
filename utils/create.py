from net import deeplab
import torch.nn.functional as F


def create_net(in_channels, num_classes, net_name):

    if net_name == 'deeplab':
        network = deeplab.DeepLab(num_classes=num_classes)
    else:
        raise ValueError('Not supported Net_name: {}'.format(net_name))
    return network


def create_loss(predicts, labels, num_classes):

    predicts = predicts.permute((0, 2, 3, 1)).reshape((-1,num_classes))
    bce_loss = F.cross_entropy(predicts, labels.flatten(), reduction='mean')
    loss = bce_loss
    return loss
