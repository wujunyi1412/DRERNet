import torch
import torchvision
import torch.nn as nn
from Net.ResNet.model import resnet34
import math

class CFM(nn.Module):
    def __init__(self, inplanes, planes):
        super(CFM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, padding=7, dilation=7),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(planes * 3, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.adp = nn.AdaptiveAvgPool2d(1)


    def forward(self, rgb, depth):
        fuse = torch.cat([rgb, depth], dim=1)
        fuse = self.conv1(fuse)
        fuse = fuse * self.adp(depth)
        rgb1 = self.conv2(rgb)
        rgb2 = self.conv3(rgb)
        rgb3 = self.conv4(rgb)
        rgb_ = torch.cat([rgb1, rgb2, rgb3], dim=1)
        rgb_ = self.conv5(rgb_)
        out = rgb_ + rgb_ * fuse

        return out


class SA(nn.Module):
    def __init__(self, inplanes, planes):
        super(SA, self).__init__()


        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, feature):
        feature_en = feature + feature * self.conv1(rgb)
        feature_en = self.conv2(feature_en)

        return feature_en




class CA(nn.Module):
    def __init__(self, inplanes, planes):
        super(CA, self).__init__()

        self.adp = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, depth, feature):
        depth_adp = self.adp(depth)
        feature_en = feature + feature * depth_adp
        feature_en = self.conv1(feature_en)

        return feature_en




class DEM5(nn.Module):
    def __init__(self, inplanes, planes):
        super(DEM5, self).__init__()

        self.ca = CA(inplanes, planes)
        self.sa = SA(inplanes, planes)

    def forward(self, rgb, depth, fuse):
        fuse_en1 = self.ca(depth, fuse)
        fuse_en2 = self.sa(rgb, fuse_en1)

        return fuse_en1, fuse_en2


class DEM(nn.Module):
    def __init__(self, inplanes, planes):
        super(DEM, self).__init__()

        self.ca = CA(planes, planes)
        self.sa = SA(planes, planes)

        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(planes * 2, planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(planes * 2, planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, depth, fuse, fuse_en1, fuse_en2):
        fuse_en1 = self.conv1(fuse_en1)
        fuse_en2 = self.conv2(fuse_en2)
        rgb = torch.cat([rgb, fuse_en2], dim=1)
        rgb = self.conv3(rgb)
        depth = torch.cat([depth, fuse_en1], dim=1)
        depth = self.conv4(depth)

        fuse_en1 = self.ca(depth, fuse)
        fuse_en2 = self.sa(rgb, fuse_en1)

        return fuse_en1, fuse_en2






class net0(nn.Module):
    def __init__(self):
        super(net0,self).__init__()

        self.backbone = resnet34(pretrained=True)

    def forward(self,depth):
        depth1 = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(depth)))
        depth1_maxpool = self.backbone.maxpool(depth1)

        depth2 = self.backbone.layer1(depth1_maxpool)
        depth3 = self.backbone.layer2(depth2)
        depth4 = self.backbone.layer3(depth3)
        depth5 = self.backbone.layer4(depth4)

        return depth1,depth2,depth3,depth4,depth5




class net1(nn.Module):
    def __init__(self):
        super(net1,self).__init__()

        self.backbone = resnet34(pretrained=True)

    def forward(self,x):
        x1 = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        x1_maxpool = self.backbone.maxpool(x1)

        x2 = self.backbone.layer1(x1_maxpool)
        x3 = self.backbone.layer2(x2)
        x4 = self.backbone.layer3(x3)
        x5 = self.backbone.layer4(x4)

        return x1,x2,x3,x4,x5

class net2(nn.Module):
    def __init__(self, dim=[64, 64, 128, 256, 512]):
        super(net2,self).__init__()

        self.net0 = net0()
        self.net1 = net1()


        self.out1 = nn.Conv2d(dim[0], 1, kernel_size=3, padding=1)
        self.out2 = nn.Conv2d(dim[1], 1, kernel_size=3, padding=1)
        self.out3 = nn.Conv2d(dim[2], 1, kernel_size=3, padding=1)
        self.out4 = nn.Conv2d(dim[3], 1, kernel_size=3, padding=1)
        self.out5 = nn.Conv2d(dim[4], 1, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        self.cfm1 = CFM(dim[0]*2, dim[0])
        self.cfm2 = CFM(dim[1]*2, dim[1])
        self.cfm3 = CFM(dim[2]*2, dim[2])
        self.cfm4 = CFM(dim[3]*2, dim[3])
        self.cfm5 = CFM(dim[4]*2, dim[4])

        self.dem5 = DEM5(dim[4], dim[4])
        self.dem4 = DEM(dim[4], dim[3])
        self.dem3 = DEM(dim[3], dim[2])
        self.dem2 = DEM(dim[2], dim[1])
        self.dem1 = DEM(dim[1], dim[0])


    def forward(self,x,depth):
        d1, d2, d3, d4, d5 = self.net0(depth)
        x1, x2, x3, x4, x5 = self.net1(x)

        fuse1 = self.cfm1(x1, d1)
        fuse2 = self.cfm2(x2, d2)
        fuse3 = self.cfm3(x3, d3)
        fuse4 = self.cfm4(x4, d4)
        fuse5 = self.cfm5(x5, d5)

        fuse5_en1, fuse5_en2 = self.dem5(x5, d5, fuse5)
        fuse4_en1, fuse4_en2 = self.dem4(x4, d4, fuse4, fuse5_en1, fuse5_en2)
        fuse3_en1, fuse3_en2 = self.dem3(x3, d3, fuse3, fuse4_en1, fuse4_en2)
        fuse2_en1, fuse2_en2 = self.dem2(x2, d2, fuse2, fuse3_en1, fuse3_en2)
        fuse1_en1, fuse1_en2 = self.dem1(x1, d1, fuse1, fuse2_en1, fuse2_en2)

        out5 = self.out5(fuse5_en2)
        out5 = self.up32(out5)

        out4 = self.out4(fuse4_en2)
        out4 = self.up16(out4)

        out3 = self.out3(fuse3_en2)
        out3 = self.up8(out3)

        out2 = self.out2(fuse2_en2)
        out2 = self.up4(out2)

        out1 = self.out1(fuse1_en2)
        out1 = self.up2(out1)

        return out1, out2, out3, out4, out5


if __name__ == "__main__":

    a = torch.randn(1, 3, 224, 224)
    b = torch.randn(1, 3, 224, 224)
    model = net2()
    from Dataprocess.FLOP import CalParams

    CalParams(model, a, b)
    print('Total params % .2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # from torchsummaryX import summary
    #
    # summary(net2, torch.zeros((1, 3, 224, 224)))


    # from toolbox import compute_speed
    from ptflops import get_model_complexity_info

    with torch.cuda.device(0):
        # net = GTLW(n_classes=9).cuda()
        net = net2()
        flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
        print('Flops:  ' + flops)
        print('Params: ' + params)








































