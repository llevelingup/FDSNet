"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from timm.models.vision_transformer import vit_base_patch16_384
from timm.models.swin_transformer import swin_base_patch4_window12_384
import torch
import torch.nn as nn
from torch.nn import init
import math
import cv2
from typing import Callable, List, Optional, Tuple, Union

class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out


# STDC2Net
class STDCNet1446(nn.Module):
    def __init__(self, base=64, layers=[4, 5, 3], block_num=4, type="cat", num_classes=1000, dropout=0.20,
                 pretrain_model='', use_conv_last=False):
        super(STDCNet1446, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = ConvX(base * 16, max(1024, base * 16), 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base * 16), max(1024, base * 16), bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base * 16))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base * 16), num_classes, bias=False)

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:6])
        self.x16 = nn.Sequential(self.features[6:11])
        self.x32 = nn.Sequential(self.features[11:])

        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):

        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(3, base // 2, 3, 2)]
        features += [ConvX(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(block(base * int(math.pow(2, i + 1)), base * int(math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(block(base * int(math.pow(2, i + 2)), base * int(math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
            feat32 = self.conv_last(feat32)

        return feat2, feat4, feat8, feat16, feat32

    def forward_impl(self, x):
        out = self.features(x)
        out = self.conv_last(out).pow(2)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        # out = self.bn(out)
        out = self.relu(out)
        # out = self.relu(self.bn(self.fc(out)))
        out = self.dropout(out)
        out = self.linear(out)
        return out


# STDC1Net
class STDCNet813(nn.Module):
    def __init__(self, base=64, layers=[2, 2, 2], block_num=4, type="cat", num_classes=1000, dropout=0.20,
                 pretrain_model='', use_conv_last=False):
        super(STDCNet813, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = ConvX(base * 16, max(1024, base * 16), 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base * 16), max(1024, base * 16), bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base * 16))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base * 16), num_classes, bias=False)

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:4])
        self.x16 = nn.Sequential(self.features[4:6])
        self.x32 = nn.Sequential(self.features[6:])

        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):

        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(3, base // 2, 3, 2)]
        features += [ConvX(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(block(base * int(math.pow(2, i + 1)), base * int(math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(block(base * int(math.pow(2, i + 2)), base * int(math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
            feat32 = self.conv_last(feat32)

        return feat2, feat4, feat8, feat16, feat32

    def forward_impl(self, x):
        out = self.features(x)
        out = self.conv_last(out).pow(2)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        # out = self.bn(out)
        out = self.relu(out)
        # out = self.relu(self.bn(self.fc(out)))
        out = self.dropout(out)
        out = self.linear(out)
        return out



class PUPHead(nn.Module):
    def __init__(self, num_classes):
        super(PUPHead, self).__init__()

        self.UP_stage_1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.UP_stage_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.UP_stage_3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        # self.UP_stage_4 = nn.Sequential(
        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # )

        self.cls_seg = nn.Conv2d(128, num_classes, 3, padding=1)

    def forward(self, x):
        x = self.UP_stage_1(x)
        x = self.UP_stage_2(x)
        x = self.UP_stage_3(x)
        # x = self.UP_stage_4(x)
        x = self.cls_seg(x)
        return x



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return  y




class MRF(nn.Module):
    def __init__(self, xbc, xsc):
        super(MRF, self).__init__()
        self.xbc = xbc
        self.xsc = xsc
        self.seb = SELayer(xbc)
        self.ses = SELayer(xsc)

        self.changeC = nn.Conv2d(xsc,xbc,kernel_size=1)
        self.linearxb = nn.Linear(xbc * xsc // 256, xbc)
        self.linearxs = nn.Linear(xbc * xsc // 256, xsc)

    def forward(self, xb, xs):
        xbc = self.xbc
        xsc = self.xsc

        tempxb = xb
        tempxs = xs


        xb = self.seb(xb)
        xs = self.ses(xs)
        xbSE = xb
        xsSE = xs
        xb = xb.view(xb.shape[0], 16, xb.shape[1] // 16)
        xs = xs.view(xs.shape[0], 16, xs.shape[1] // 16)
        matrixOut = torch.bmm(xs.permute(0,2,1),xb).flatten(1)
        outB = torch.sigmoid(self.linearxb(matrixOut).unsqueeze(2).unsqueeze(3))
        outS = torch.sigmoid(self.linearxs(matrixOut).unsqueeze(2).unsqueeze(3))

        outS = outS+xsSE
        outB = outB+xbSE

        xbOut = outB*tempxb
        xsOut = outS*tempxs

        xsOut = self.changeC(xsOut)
        xsOut = F.interpolate(xsOut, scale_factor=2, mode='bilinear', align_corners=False)
        return xbOut+xsOut



class backbone(nn.Module):
    def __init__(self, num_classes):
        super(backbone, self).__init__()
        self.backbone = STDCNet813(num_classes=104,dropout=0.2)

    def forward(self, x):
        x = self.backbone(x)
        return x




class Backbone1(nn.Module):
    def __init__(self, num_classes,norm_layer: Union[str, Callable] = nn.LayerNorm):
        super(Backbone1, self).__init__()
        self.deepNet = swin_base_patch4_window12_384(pretrained = True)
        self.sixTothree = nn.Conv2d(6,3,kernel_size=1)
        # self.cp就是浅层网络
        self.cp = backbone(num_classes=num_classes)
        self.inputdown2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.MRF0 = MRF(512,1024)
        self.MRF1 = MRF(512,512)
        self.MRF2 = MRF(256,512)
        self.PuPHead = PUPHead(104)
        self.norm = norm_layer(512)
    def forward(self, x):
        device = x.device
        x_np = x.cpu().numpy().transpose(0, 2, 3, 1)
        # x_np = x_np.to(device)
        laplacian_layers = []
        laplacian_layers2 = []
        batch_size = x_np.shape[0]
        for i in range(batch_size):
            # 构建高斯金字塔的第一层
            gaussian_down = cv2.pyrDown(x_np[i])
            gaussian_down2 = cv2.pyrDown(gaussian_down)
            # 将下采样的图像上采样回原始大小
            gaussian_up = cv2.pyrUp(gaussian_down, dstsize=(x_np[i].shape[1], x_np[i].shape[0]))
            gaussian_up2 = cv2.pyrUp(gaussian_down2, dstsize=(gaussian_down.shape[1],gaussian_down.shape[0]))

            # 计算拉普拉斯图像
            laplacian = cv2.subtract(x_np[i], gaussian_up)
            laplacian2 = cv2.subtract(gaussian_down,gaussian_up2)


            # 将结果转换回 PyTorch 张量并添加到列表中
            laplacian_tensor = torch.from_numpy(laplacian.transpose(2, 0, 1))  # 转换回 [channels, H, W]
            laplacian2_tensor = torch.from_numpy(laplacian2.transpose(2, 0, 1))

            laplacian_layers.append(laplacian_tensor)
            laplacian_layers2.append(laplacian2_tensor)
            # 将列表转换为张量，维度为 [batch_size, channels, H, W]
        laplacian_layers = torch.stack(laplacian_layers, dim=0).to(device)
        laplacian_layers2 = torch.stack(laplacian_layers2, dim=0).to(device)
        shallowInput = []
        for i in range(batch_size):

            laplacian_layers2_Temp = laplacian_layers2[i].unsqueeze(0)
            laplacian_layers2_Temp =F.interpolate(laplacian_layers2_Temp, scale_factor=2, mode='bilinear',align_corners=False)
            laplacian_layers2_Temp = laplacian_layers2_Temp.squeeze(0)
            shallowInput.append(torch.cat((laplacian_layers[i],laplacian_layers2_Temp),dim=0))
        shallowInput = torch.stack(shallowInput, dim=0)
        shallowInput = self.sixTothree(shallowInput)

        shallowX = self.cp(shallowInput)

        deepX = self.inputdown2(x)
        deepX = self.deepNet(deepX)
        deepX[0] = deepX[0].permute(0,2,3,1)
        deepX[0] = self.norm(deepX[0])
        deepX[0] = deepX[0].permute(0,3,1,2)
        MRFout0 = self.MRF0(deepX[0],deepX[1])
        MRFout1 = self.MRF1(shallowX[3],MRFout0)
        result = self.MRF2(shallowX[2],MRFout1)
        result = self.PuPHead(result)

        return result

def FDSNet(num_classes: int = 104, has_logits: bool = True):

    model = Backbone1(num_classes=104)
    return model






if __name__ == "__main__":
    # VIT-Large  设置了16个patch
    SETRNet = FDSNet(num_classes=104)
    img = torch.randn(2, 3, 768, 768)
    preds = SETRNet(img)
    print(preds.shape)