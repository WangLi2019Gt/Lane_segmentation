import torch
import config
import utils
import pandas as pd
import os
import numpy as np
from torch.hub import load_state_dict_from_url
import cv2
from os.path import join as pjoin
if __name__ == '__main__':
    cfg=config.ConfigTest()
    print("Device: ", cfg.DEVICE)
    device = torch.device(cfg.DEVICE)

    print("Net: ", cfg.NET_NAME)
    #net = utils.create_net(cfg.IN_CHANNEL, cfg.NUM_CLASSES, cfg.NET_NAME).cuda()
    net = utils.create_net(cfg.IN_CHANNEL, cfg.NUM_CLASSES, cfg.NET_NAME, cfg.BACKBONE).cuda()
    if cfg.WEIGHTS:
        print('load weights from: ', cfg.WEIGHTS)
        net.load_state_dict_from_url(torch.load(cfg.WEIGHTS))
    else:
        print("fuck, no weight")
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.BASE_LR)
    print('Prepare data...batch_size: {}, img_size: {}, crop_offset: {}'.format(cfg.BATCH_SIZE, cfg.IMG_SIZE, cfg.CROP_OFFSET))
    df_test = pd.read_csv(os.path.join(cfg.DATA_LIST_DIR,'test.csv'))
    data_generator = utils.train_data_generator(np.array(df_test['image']),
                                                None,
                                                cfg.BATCH_SIZE, cfg.IMG_SIZE, cfg.CROP_OFFSET,file_path=True)


    print('Begin infenrence...')
    done_num=0
    while True:
        imgs,file = next(data_generator)
        if imgs is None:
            break
        imgs = imgs.to(device)
        predicts = net(imgs).cpu().detach().numpy()
        outs = utils.decodePredicts(predicts, cfg.IMAGE_SIZE_ORG, cfg.CROP_OFFSET, mode='grey')

        # 保存
        for i, out in enumerate(outs):
            cv2.imwrite(pjoin(cfg.LABEL_ROOT, file[i].split("/")[-1].replace('.jpg', '_bin.png')), out)
            org_image = cv2.imread(pjoin(cfg.IMAGE_ROOT, file[i].split("/")[-1]))
            overlay_image = cv2.addWeighted(org_image, 0.6, out, 0.4, gamma=0)
            cv2.imwrite(pjoin(cfg.OVERLAY_ROOT, file[i].split("/")[-1].replace('.jpg', '.png')), overlay_image)

        done_num += imgs.shape[0]
        print('Finished {} images'.format(done_num))

    print('Done')








