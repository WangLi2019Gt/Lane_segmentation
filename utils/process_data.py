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
def train_data_generator(imgs, labels, batch_size, img_size, crop_offset):
    batch_index = np.arange(0, len(labels))
    img_out=[]
    label_out=[]
    while True:
        np.random.shuffle(batch_index)
        for i in batch_index:
            if os.path.exist(imgs[i]):
                img = cv2.imread(imgs[i])
                label = cv2.imread(labels[i], cv2.IMREAD_GRAYSCALE)

                train_img, train_label= crop_resize(img, label, img_size, crop_offset)

                train_label = lp.encode_labels(train_label)

                img_out.append(train_img)
                label_out.append(train_label)
                if len(img_out)>=batch_size:
                    img_out=torch.from_numpy(np.array(img_out))
                    label_out=torch.from_numpy(np.array(label_out))
                    img_out=img_out[:, :, :, ::-1].transpose(0, 3, 1, 2).float() / (255.0 / 2) - 1
                    label_out = label_out.long()
                    yield img_out, label_out
                    img_out, label_out=[], []
            else:
                print(imgs[i], 'not exist')