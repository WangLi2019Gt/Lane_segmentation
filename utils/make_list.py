import os
import pandas as pd
from sklearn.utils import shuffle
img_list=[]
lab_list=[]
img_dir='/home/rogqigx171/data/ColorImage_road02/ColorImage/'
lab_dir='/home/rogqigx171/data/Gray_Label/Label_road02/Label/'

for d1 in os.listdir(img_dir):
    img_sub_dir1=os.path.join(img_dir, d1)
    lab_sub_dir1=os.path.join(lab_dir, d1)
    for d2 in os.listdir(img_sub_dir1):
        img_sub_dir2=os.path.join(img_sub_dir1, d2)
        lab_sub_dir2=os.path.join(lab_sub_dir1, d2)
        for d3 in os.listdir(img_sub_dir2):
            label_d3 = d3.replace('.jpg','_bin.png')
            img_sub_dir3=os.path.join(img_sub_dir2, d3)
            lab_sub_dir3=os.path.join(lab_sub_dir2, label_d3)
			
            if not os.path.exists(img_sub_dir3):
                print(img_sub_dir3) 
                continue
            if not os.path.exists(lab_sub_dir3):
                print(lab_sub_dir3)
                continue
            img_list.append(img_sub_dir3)
            lab_list.append(lab_sub_dir3)

assert len(img_list) == len(lab_list)

all = pd.DataFrame({'image': img_list, 'label':lab_list})
all_shuffle = shuffle(all)

#sixth_part = int(len(img_list) * 0.6)
#eighth_part = int(len(img_list) * 0.8)
sixth_part = int(120)
eighth_part=int(160)

train_dataset = all_shuffle[:sixth_part]
val_dataset = all_shuffle[sixth_part:eighth_part]
test_dataset = all_shuffle[eighth_part:int(eighth_part*1.25)]

train_dataset.to_csv('/home/rogqigx171/Lane_segmentation/data_list/train.csv',index=False)
val_dataset.to_csv('/home/rogqigx171/Lane_segmentation/data_list/val.csv', index=False)
test_dataset.to_csv('/home/rogqigx171/Lane_segmentation/data_list/test.csv', index=False)
