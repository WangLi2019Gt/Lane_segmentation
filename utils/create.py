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

def ajust_learning_rate(optimizer, lr_strategy, epoch, iteration, epoch_size):
    """
    根据给定的策略调整学习率
    @param optimizer: 优化器
    @param lr_strategy: 策略，一个二维数组，第一维度对应epoch，第二维度表示在一个epoch内，若干阶段的学习率
    @param epoch: 当前在第几号epoch
    @param iteration: 当前epoch内的第几次迭代
    @param epoch_size: 当前epoch的总迭代次数
    """
    assert epoch < len(lr_strategy), 'lr strategy unconvering all epoch'
    batch = epoch_size // len(lr_strategy[epoch])
    lr = lr_strategy[epoch][-1]
    for i in range(len(lr_strategy[epoch])):
        if iteration < (i + 1) * batch:
            lr = lr_strategy[epoch][i]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            break
    return lr

