import torch
import torch.nn as nn
import math
from torch.hub import load_state_dict_from_url
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)  # ? Why no bias: 如果卷积层之后是BN层，那么可以不用偏置参数，可以节省内存
def GroupNorm(out_planes):
    return nn.GroupNorm(32,out_planes)
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, groups=1,base_width=64,stride=1, dilation=1, downsample=None, norm_layer=None,SE=False):
        super(SEBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, groups=groups, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se=SE
        if SE:
            self.selayer=SELayer(planes*self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.se:
            out=self.selayer(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEResNeXt(nn.Module):
    def __init__(self, block, layers, output_stride, groups=1,width_per_group=64 ,norm_layer=None,SE=False):
        super(SEResNeXt, self).__init__()
        blocks = [1, 2, 4]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        else:
            raise NotImplementedError
        self.SE=SE
        self.inplanes = 64
        self.base_width=width_per_group
        self.groups=groups
        # conv1 in ppt figure
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3])
        self._init_weight()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        # 生成不同的stage/layer
        # block: block type(basic block/bottle block)
        # blocks: blocks的数量
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            # 需要调整维度
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),  # 同时调整spatial(H x W))和channel两个方向
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes,self.groups,self.base_width, stride, dilation, downsample, norm_layer,self.SE))  # 第一个block单独处理
        self.inplanes = planes * block.expansion  # 记录layerN的channel变化，具体请看ppt resnet表格
        for _ in range(1, blocks):  # 从1开始循环，因为第一个模块前面已经单独处理
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=dilation, norm_layer=norm_layer,SE=self.SE))
        return nn.Sequential(*layers)  # 使用Sequential层组合blocks，形成stage。如果layers=[2,3,4]，那么*layers=？

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        norm_layer = self._norm_layer
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,self.groups,self.base_width, stride, dilation=blocks[0] * dilation,
                            downsample=downsample, norm_layer=norm_layer,SE=self.SE))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width,stride=1,
                                dilation=blocks[i] * dilation, norm_layer=norm_layer,SE=self.SE))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def _resnet(arch, block, layers, output_stride, pretrained, progress, **kwargs):

    model = SEResNeXt(block, layers, output_stride, **kwargs)
    if pretrained:
        pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/resnext101_32x4d-29e315fa.pth')
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
    return model


def resneXt101_32_8d(output_stride, pretrained=True, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    kwargs['SE'] = False
    return _resnet('resneXt101_32_4d', SEBottleneck, [3, 4, 23, 3], output_stride, pretrained, progress,
                   **kwargs)
def SEresneXt101_32_4d(output_stride, pretrained=True, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    kwargs['SE'] = True
    return _resnet('resneXt101_32_4d', SEBottleneck, [3, 4, 23, 3], output_stride, pretrained, progress,
                   **kwargs)

def SEresneXt101_32_4d_GN(output_stride, pretrained=True, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    kwargs['SE'] = True
    kwargs['norm_layer'] = GroupNorm
    return _resnet('resneXt101_32_4d', SEBottleneck, [3, 4, 23, 3], output_stride, pretrained, progress,
                   **kwargs)

if __name__ == '__main__':
    model = SEresneXt101_32_4d(16,pretrained=True)
    print(model.eval())
    
