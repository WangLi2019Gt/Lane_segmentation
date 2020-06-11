import os
import torch

class ConfigTrain(object):
    def __init__(self):
        self.DEVICE ='cuda:0'
        self.NET_NAME='deeplab'
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
