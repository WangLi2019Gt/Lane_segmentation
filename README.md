# LANE SEGMENTATION

### Methods 
设备：单张NVIDIA TESLA T4， 16GB显存
输入：3384x1710的车道线图片，裁掉上部分天空等与车道线无关图片（690高）数据集（BAIDU ROAD02）
因显存有限，对剩余3384x1020图片进行放缩。目前使用768*256，1024x384,1536x512三种分辨率；下一步尝试完全等比例放缩
模型：以SE-RESNEXT为backbone的deeplabv3+模型,其中Resnext部分载入预训练参数。下一步，新增UNET
LOSS: CE LOSS+DICE LOSS
训练策略：基于小分辨率预训练，在小分辨率基础上进行大分辨率训练
Cycle的分步学习率策略，ADAM作为优化器，训练8个EPOCH，以0.001作为BASE_LR
输出：分辨率放缩至3384x1020

### Result of Deeplabv3+

| backbone | loss  | Base LR |Batch Size|Resolution|Miou|log|
| :--------- |:------------:| -----:|-----:|-----:|-----:|-----:|
| SE-ResneXt101_32_4d | bce + dice | 0.001 |8|768 x 256|0.554|75.log
| SE-ResneXt101_32_4d    | bce + dice     |   0.001 |4|1024 x 384|0.571|76.log|
| SE-ResneXt101_32_4d  | bce + dice   |  0.001|2|1536 x 512|0.528|76_1.log|
| SE-ResneXt101_32_4d_GN|   bce + dice       |   0.001 | 8 | 768 x 256        |    0.539 ||
| SE-ResneXt101_32_4d_GN|   bce + dice       |    0.001 | 4 |1024 x 384	      |    0.558 |78_GN_1152_384.log |
| SE-ResneXt101_32_4d_GN|  bce + dice  |    0.001 |2  |1536 x 512   | 0.526 |78_GN_1536_512.log|

日志存在./logs文件夹中。
问题：分辨率增到1536x512后出现明显的性能下降，最初我认为是因为BATCHSIZE只有2，导致BN效果不好，换用了GN，结果没什么改观。
我的训练策略是每次采用上一个分辨率最终的模型作为预加载的模型进行训练。是否因此而过拟合，但是日志中train miou和val miou基本差不多。下一步打算调整下学习率策略，并且简化模型或使用UNET进行尝试

SE-ResneXt101_32_4d weights: https://drive.google.com/file/d/18S5r0ipPa0AqZzl-dY0_NM8ui4SsEPAF/view?usp=sharing

### Example of Inference
![Image text](https://raw.githubusercontent.com/WangLi2019Gt/Lane_segmentation/master/img/170927_064216654_Camera_6.jpg)

![Image text](https://raw.githubusercontent.com/WangLi2019Gt/Lane_segmentation/master/img/170927_064216654_Camera_6_bin.png)

### How to use
pip install -i https://test.pypi.org/simple/ wang==1.2.0

from Lane_segmentation import inference

inference.infer("/home/rogqigx171/Lane_segmentation/weights_1/ep_7_ls_0.021_0.570.pth","/home/rogqigx171/Lane_segmentation/data_list/test.csv","/home/rogqigx171/out")

#inference.infer(weight_dir,test_csv_dir,output_dir)
