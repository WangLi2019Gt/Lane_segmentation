import os
import torch
from os.path import join as pjoin
from os.path import dirname, abspath
import net.se_resnext
class ConfigTrain(object):
    def __init__(self):
        self.DEVICE ='cuda:0'
        self.NET_NAME='deeplab'
        self.BACKBONE = net.se_resnext.SEresneXt101_32_4d(16)
        self.WEIGHTS=None
        self.IN_CHANNEL=3
        self.NUM_CLASSES = 8
        self.BASE_LR=0.006
        self.DATA_LIST_DIR='./data_list'
        self.LOSS = 'ce'
        self.BATCH_SIZE=2
        self.IMG_SIZE=(1024,384)
        self.CROP_OFFSET=690

        self.LOG_DIR='./logs'
        self.EPOCH_BEGIN=0
        self.EPOCH_NUM=10
        self.SAVE_DIR='./weights'

class ConfigTest(object):
    def __init__(self):
        self.DEVICE = 'cuda:0'
        self.NET_NAME = 'deeplab'
        self.BACKBONE = net.se_resnext.SEresneXt101_32_4d(16)
        self.WEIGHTS = None
        self.IN_CHANNEL = 3
        self.NUM_CLASSES = 8
        self.BASE_LR = 0.006
        self.DATA_LIST_DIR = './data_list'
        self.LOSS = 'ce'
        self.BATCH_SIZE = 2
        self.IMG_SIZE = (1024, 384)
        self.CROP_OFFSET = 690

        self.LOG_DIR = './logs'
        self.EPOCH_BEGIN = 0
        self.EPOCH_NUM = 10
        self.SAVE_DIR = './weights'
        self.IMAGE_SIZE_ORG = (3384, 1710)
        self.PROJECT_ROOT = dirname(abspath(__file__))
        self.DATA_ROOT = pjoin(self.PROJECT_ROOT, 'test')
        self.IMAGE_ROOT = pjoin(self.DATA_ROOT, 'TestImage')
        self.LABEL_ROOT = pjoin(self.DATA_ROOT, 'label')
        self.OVERLAY_ROOT = pjoin(self.DATA_ROOT, 'overlay')
