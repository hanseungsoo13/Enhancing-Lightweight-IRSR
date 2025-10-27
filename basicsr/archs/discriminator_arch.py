from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from basicsr.utils.registry import ARCH_REGISTRY
import basicsr.archs.basicblock as B


@ARCH_REGISTRY.register()
class VGGStyleDiscriminator(nn.Module):
    """VGG style discriminator with input size 128 x 128 or 256 x 256.

    It is used to train SRGAN, ESRGAN, and VideoGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.Default: 64.
    """

    def __init__(self, num_in_ch, num_feat, input_size=128):
        super(VGGStyleDiscriminator, self).__init__()
        self.input_size = input_size
        assert (
            self.input_size == 128 or self.input_size == 256
        ), f"input size must be 128 or 256, but received {input_size}"

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        if self.input_size == 256:
            self.conv5_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
            self.bn5_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
            self.conv5_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
            self.bn5_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == self.input_size, f"Input size must be identical to input_size, but received {x.size()}."

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: /2

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: /4

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: /8

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: /16

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: /32

        if self.input_size == 256:
            feat = self.lrelu(self.bn5_0(self.conv5_0(feat)))
            feat = self.lrelu(self.bn5_1(self.conv5_1(feat)))  # output spatial size: / 64

        # spatial size: (4, 4)
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out


@ARCH_REGISTRY.register(suffix="basicsr")
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


"""
# --------------------------------------------
# Discriminator_VGG_96
# Discriminator_VGG_128
# Discriminator_VGG_192
# Discriminator_VGG_128_SN
# --------------------------------------------
"""


# --------------------------------------------
# VGG style Discriminator with 96x96 input
# --------------------------------------------
@ARCH_REGISTRY.register()
class Discriminator_VGG_96(nn.Module):
    def __init__(self, in_nc=3, base_nc=64, ac_type="BL"):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = B.conv(in_nc, base_nc, kernel_size=3, mode="C")
        conv1 = B.conv(base_nc, base_nc, kernel_size=4, stride=2, mode="C" + ac_type)
        # 48, 64
        conv2 = B.conv(base_nc, base_nc * 2, kernel_size=3, stride=1, mode="C" + ac_type)
        conv3 = B.conv(base_nc * 2, base_nc * 2, kernel_size=4, stride=2, mode="C" + ac_type)
        # 24, 128
        conv4 = B.conv(base_nc * 2, base_nc * 4, kernel_size=3, stride=1, mode="C" + ac_type)
        conv5 = B.conv(base_nc * 4, base_nc * 4, kernel_size=4, stride=2, mode="C" + ac_type)
        # 12, 256
        conv6 = B.conv(base_nc * 4, base_nc * 8, kernel_size=3, stride=1, mode="C" + ac_type)
        conv7 = B.conv(base_nc * 8, base_nc * 8, kernel_size=4, stride=2, mode="C" + ac_type)
        # 6, 512
        conv8 = B.conv(base_nc * 8, base_nc * 8, kernel_size=3, stride=1, mode="C" + ac_type)
        conv9 = B.conv(base_nc * 8, base_nc * 8, kernel_size=4, stride=2, mode="C" + ac_type)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --------------------------------------------
# VGG style Discriminator with 128x128 input
# --------------------------------------------
@ARCH_REGISTRY.register()
class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc=3, base_nc=64, ac_type="BL"):
        super(Discriminator_VGG_128, self).__init__()
        # features
        # hxw, c
        # 128, 64
        conv0 = B.conv(in_nc, base_nc, kernel_size=3, mode="C")
        conv1 = B.conv(base_nc, base_nc, kernel_size=4, stride=2, mode="C" + ac_type)
        # 64, 64
        conv2 = B.conv(base_nc, base_nc * 2, kernel_size=3, stride=1, mode="C" + ac_type)
        conv3 = B.conv(base_nc * 2, base_nc * 2, kernel_size=4, stride=2, mode="C" + ac_type)
        # 32, 128
        conv4 = B.conv(base_nc * 2, base_nc * 4, kernel_size=3, stride=1, mode="C" + ac_type)
        conv5 = B.conv(base_nc * 4, base_nc * 4, kernel_size=4, stride=2, mode="C" + ac_type)
        # 16, 256
        conv6 = B.conv(base_nc * 4, base_nc * 8, kernel_size=3, stride=1, mode="C" + ac_type)
        conv7 = B.conv(base_nc * 8, base_nc * 8, kernel_size=4, stride=2, mode="C" + ac_type)
        # 8, 512
        conv8 = B.conv(base_nc * 8, base_nc * 8, kernel_size=3, stride=1, mode="C" + ac_type)
        conv9 = B.conv(base_nc * 8, base_nc * 8, kernel_size=4, stride=2, mode="C" + ac_type)
        # 4, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --------------------------------------------
# VGG style Discriminator with 192x192 input
# --------------------------------------------
@ARCH_REGISTRY.register()
class Discriminator_VGG_192(nn.Module):
    def __init__(self, in_nc=3, base_nc=64, ac_type="BL"):
        super(Discriminator_VGG_192, self).__init__()
        # features
        # hxw, c
        # 192, 64
        conv0 = B.conv(in_nc, base_nc, kernel_size=3, mode="C")
        conv1 = B.conv(base_nc, base_nc, kernel_size=4, stride=2, mode="C" + ac_type)
        # 96, 64
        conv2 = B.conv(base_nc, base_nc * 2, kernel_size=3, stride=1, mode="C" + ac_type)
        conv3 = B.conv(base_nc * 2, base_nc * 2, kernel_size=4, stride=2, mode="C" + ac_type)
        # 48, 128
        conv4 = B.conv(base_nc * 2, base_nc * 4, kernel_size=3, stride=1, mode="C" + ac_type)
        conv5 = B.conv(base_nc * 4, base_nc * 4, kernel_size=4, stride=2, mode="C" + ac_type)
        # 24, 256
        conv6 = B.conv(base_nc * 4, base_nc * 8, kernel_size=3, stride=1, mode="C" + ac_type)
        conv7 = B.conv(base_nc * 8, base_nc * 8, kernel_size=4, stride=2, mode="C" + ac_type)
        # 12, 512
        conv8 = B.conv(base_nc * 8, base_nc * 8, kernel_size=3, stride=1, mode="C" + ac_type)
        conv9 = B.conv(base_nc * 8, base_nc * 8, kernel_size=4, stride=2, mode="C" + ac_type)
        # 6, 512
        conv10 = B.conv(base_nc * 8, base_nc * 8, kernel_size=3, stride=1, mode="C" + ac_type)
        conv11 = B.conv(base_nc * 8, base_nc * 8, kernel_size=4, stride=2, mode="C" + ac_type)
        # 3, 512
        self.features = B.sequential(
            conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11
        )

        # classifier
        self.classifier = nn.Sequential(nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --------------------------------------------
# SN-VGG style Discriminator with 128x128 input
# --------------------------------------------
@ARCH_REGISTRY.register()
class Discriminator_VGG_128_SN(nn.Module):
    def __init__(self):
        super(Discriminator_VGG_128_SN, self).__init__()
        # features
        # hxw, c
        # 128, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv1 = spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        # 64, 64
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        # 32, 128
        self.conv4 = spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        # 16, 256
        self.conv6 = spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 8, 512
        self.conv8 = spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 4, 512

        # classifier
        self.linear0 = spectral_norm(nn.Linear(512 * 4 * 4, 100))
        self.linear1 = spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x
