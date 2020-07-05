import numpy as np
import os
import cv2
import utils.label_process as lp
import torch

def crop_resize(img, label=None, img_size=(1024,384), crop_offset=690):
    roi = img[crop_offset:,:]
    train_img = cv2.resize(roi, (img_size[0], img_size[1]), interpolation=cv2.INTER_LINEAR)
    if label is not None:
        roi_label = label[crop_offset:,:]
        train_label = cv2.resize(roi_label, (img_size[0], img_size[1]), interpolation=cv2.INTER_NEAREST)
        return train_img, train_label
    else:
        return train_img
def train_data_generator(imgs, labels=None, batch_size=2, img_size=(1024, 384), crop_offset=690,file_path=False):
    batch_index = np.arange(0, imgs.shape[0])
    img_out=[]
    filep=[]
    if labels is not None:
        label_out=[]
    while True:
        np.random.shuffle(batch_index)
        for i in batch_index:
            if os.path.exists(imgs[i]):
                img = cv2.imread(imgs[i])
                if  labels:
                    label = cv2.imread(labels[i], cv2.IMREAD_GRAYSCALE)
                    train_img, train_label= crop_resize(img, label, img_size, crop_offset)
                    train_label = lp.encode_labels(train_label)
                    label_out.append(train_label)
                else:
                    train_img= crop_resize(img, None, img_size, crop_offset)
                img_out.append(train_img)
                filep.append(imgs[i])
                if len(img_out)>=batch_size:
                    img_out=torch.from_numpy(np.array(img_out))
                    img_out = img_out.permute(0, 3, 1, 2).float() / (255.0 / 2) - 1
                    if labels:
                        label_out=torch.from_numpy(np.array(label_out))
                        label_out = label_out.long()
                        yield img_out, label_out
                        img_out, label_out=[], []
                    elif file_path:
                        yield img_out, filep
                        img_out=[]
                        filep=[]
                    else:
                        yield img_out
                        img_out = []
            else:
                print(imgs[i], 'not exist')

def decodePredicts(predicts, out_size, height_pad_offset, mode='gray'):
    """
    将推断的结果恢复成图片
    @param predicts: shape=(n, c, h, w)
    @param out_size: 恢复的尺寸 (w, h)
    @param height_pad_offset: 在高度维度上填充回多少
    @param mode: color | gray
    """
    # softmax
    predicts = np.argmax(predicts, axis=1)
    # reshape to (n, -1)
    n, h, w = predicts.shape
    predicts = predicts.reshape((n, -1))
    if mode == 'color':
        predicts = lp.decode_color_labels(predicts)
        predicts = predicts.reshape((3, n, h, w))
        predicts = predicts.transpose((1, 2, 3, 0)) # to (n, h, w, c)
        c = 3
    elif mode == 'gray':
        predicts = lp.decode_labels(predicts)
        predicts=predicts.reshape((n, 1, h, w))
        print(predicts.shape)
        predicts = predicts.transpose((0, 2, 3, 1)) # to (n, h, w, c)
        c = 1
    else:
        raise ValueError('mode supports: color / gary')

    # resize & pad (必须用最近邻)
    dsize = (out_size[0], out_size[1]-height_pad_offset)
    outs = []
    for i in range(n):
        out = np.zeros((out_size[1], out_size[0]), dtype=np.uint8)
        out[height_pad_offset:] = cv2.resize(predicts[i], dsize, interpolation=cv2.INTER_NEAREST)  # label
        outs.append(out)
    return outs
