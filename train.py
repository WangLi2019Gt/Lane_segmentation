import torch
import config
import utils
import pandas as pd
import os
import numpy as np
from torch.hub import load_state_dict_from_url

if __name__ == '__main__':
    cfg=config.ConfigTrain()
    print("Device: ", cfg.DEVICE)
    device = torch.device(cfg.DEVICE)

    print("Net: ", cfg.NET_NAME)
    net = utils.create_net(cfg.IN_CHANNEL, cfg.NUM_CLASSES, cfg.NET_NAME).cuda()
    if cfg.WEIGHTS:
        print('load weights from: ', cfg.WEIGHTS)
        net.load_state_dict_from_url(torch.load(cfg.WEIGHTS))
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.BASE_LR)
    print('Prepare data...batch_size: {}, img_size: {}, crop_offset: {}'.format(cfg.BATCH_SIZE, cfg.IMG_SIZE, cfg.CROP_OFFSET))
    df_train = pd.read_csv(os.path.join(cfg.DATA_LIST_DIR,'train.csv'))
    data_generator = utils.train_data_generator(np.array(df_train['image']),
                                                np.array(df_train['label']),
                                                cfg.BATCH_SIZE, cfg.IMG_SIZE, cfg.CROP_OFFSET)

    df_val= pd.read_csv(os.path.join(cfg.DATA_LIST_DIR, 'val.csv'))
    val_data_generator = utils.train_data_generator(np.array(df_val['image']),
                                                np.array(df_val['label']),
                                                cfg.BATCH_SIZE, cfg.IMG_SIZE, cfg.CROP_OFFSET)
    print('Begin training...')
    log_iter=1
    batch_num=int(len(df_train)/cfg.BATCH_SIZE)
    for epoch in range(cfg.EPOCH_BEGIN,cfg.EPOCH_NUM):
        epoch_loss=0.0
        epoch_miou = 0.0
        last_epoch_loss=0.0
        net.train()
        for iter in range(1,batch_num+1):
            imgs, labels= next(data_generator)
            imgs = imgs.to(device)
            labels = labels.to(device)

            predicts = net(imgs)
            optimizer.zero_grad()

            loss, miou = utils.create_loss(predicts, labels, cfg.NUM_CLASSES)
            epoch_loss+=loss.item()
            epoch_miou += miou.item()
            print("TRAIN:[epoch-%d , iter-%d]  iter loss:%.3f  iter miou:%.3f  epoch_loss:%.3f  epoch_miou:%.3f" % (
                epoch, iter, loss.item(), miou.item(),epoch_loss/iter,epoch_miou/iter))

            loss.backward()
            optimizer.step()
        #val
        val_loss = 0.0
        val_miou = 0.0
        net.eval()
        with torch.no_grad():
            for iter in range(1, batch_num + 1):
                imgs, labels = next(val_data_generator)
                imgs = imgs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                predicts = net(imgs)
                loss, miou = utils.create_loss(predicts, labels, cfg.NUM_CLASSES)
                val_loss += loss.item()
                val_miou += miou.item()
                print("EVAL:[epoch-%d , iter-%d]  iter loss:%.3f  iter miou:%.3f  epoch_loss:%.3f  epoch_miou:%.3f" % (
                    epoch, iter, loss.item(), miou.item(), val_loss / iter, val_miou / iter))

        torch.save(net.state_dict(),os.path.join(cfg.SAVE_DIR,"ep_%d_ls_%.3f.pth" % (epoch,epoch_loss/batch_num)))





