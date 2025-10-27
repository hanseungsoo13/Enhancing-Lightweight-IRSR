import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

import torchvision.transforms as transforms
from basicsr.ops.open_clip import clip

_reduction_modes = ["none", "mean", "sum"]


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction="none")


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction="none")


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target) ** 2 + eps)


@LOSS_REGISTRY.register()
class L_clip(nn.Module):
    def __init__(self):
        super(L_clip, self).__init__()

        local_rank = torch.distributed.get_rank()
        self.device = f"cuda:{local_rank}"

        # for clip reconstruction loss
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, download_root="./clip_model/")
        self.model.eval()
        for para in self.model.parameters():
            para.requires_grad = False

    def get_clip_score(self, tensor, words):
        score = 0
        for i in range(tensor.shape[0]):
            # image preprocess
            clip_normalizer = transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
            )
            img_resize = transforms.Resize((224, 224))
            image2 = img_resize(tensor[i])
            image = clip_normalizer(image2).unsqueeze(0)
            # get probabilitis
            text = clip.tokenize(words).to(self.device)
            with torch.no_grad():
                logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1)
            # 2-word-compared probability
            prob = probs[0][0] / probs[0][1]  # you may need to change this line for more words comparison
            # prob = probs[0][0]
            score = score + prob

        return score / tensor.shape[0]

    def forward(self, x):
        # k1 = get_clip_score(x,["A low resolution photo","A high resolution photo"])
        k2 = self.get_clip_score(x, ["A blurry and dim photo", "A clear and sharp photo"])
        # k3 = get_clip_score(x,["A low-contrast, detail-missing, and blurry photo","A high-contrast, clear, and textured-rich photo"])
        k = k2
        return 0.2 * k


@LOSS_REGISTRY.register()
class L_clip_MSE(nn.Module):
    def __init__(self):
        super(L_clip_MSE, self).__init__()
        local_rank = torch.distributed.get_rank()
        device = f"cuda:{local_rank}"

        # for clip reconstruction loss
        self.res_model, res_preprocess = clip.load("RN101", device=device, download_root="./clip_model/")
        self.res_model.eval()
        for para in self.res_model.parameters():
            para.requires_grad = False

    def get_clip_score_MSE(self, pred, inp, weight):
        score = 0
        for i in range(pred.shape[0]):
            clip_normalizer = transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
            )
            img_resize = transforms.Resize((224, 224))
            pred_img = img_resize(pred[i])
            pred_img = clip_normalizer(pred_img.reshape(1, 3, 224, 224))
            pred_image_features = self.res_model.encode_image(pred_img)

            inp_img = img_resize(inp[i])
            inp_img = clip_normalizer(inp_img.reshape(1, 3, 224, 224))
            inp_image_features = self.res_model.encode_image(inp_img)

            MSE_loss_per_img = 0
            for feature_index in range(len(weight)):
                MSE_loss_per_img = MSE_loss_per_img + weight[feature_index] * F.mse_loss(
                    pred_image_features[0][feature_index].squeeze(0), inp_image_features[0][feature_index].squeeze(0)
                )
                score = score + MSE_loss_per_img
        return score

    def forward(self, pred, inp, weight=[1.0, 1.0, 1.0, 1.0, 0.5]):
        res = self.get_clip_score_MSE(pred, inp, weight)
        return res


@LOSS_REGISTRY.register()
class ContrasLoss(nn.Module):
    def __init__(self, d_func="L1", loss_weight=1.0, weights=[]):
        super(ContrasLoss, self).__init__()
        self.d_func = d_func
        self.loss_weight = loss_weight
        self.weights = weights

    def forward(self, latents):
        if self.d_func == "L1":
            self.forward_func = self.L1_forward

        return self.forward_func(latents)

    def L1_forward(self, latents):
        """
        :param latents: n*(1+1+neg_num)*batch*token*dim
        :param anc: batch*token*dim
        :param pos: batch*token*dim
        "param negs: neg_num*batch*token*dim
        """

        loss = 0
        for i in range(len(latents)):
            anc = latents[i][0]
            pos = latents[i][1]
            d_ap = torch.mean(torch.abs(anc - pos))

            negs = torch.stack(latents[i][2:])
            d_an = torch.mean(torch.abs(anc - negs).sum(0))

            contras = d_ap / (d_an + 1e-7)
            loss += self.weights[i] * contras

        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class CELoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(CELoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}")

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C). Predicted tensor.
            target (Tensor): of shape (N, C). Ground truth tensor.
        """
        return self.loss_weight * F.cross_entropy(pred, target, reduction=self.reduction)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(L1Loss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}")

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(MSELoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}")

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction="mean", eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}")

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        if reduction not in ["mean", "sum"]:
            raise ValueError(f"Unsupported reduction mode: {reduction}. Supported ones are: mean | sum")
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(
        self,
        layer_weights,
        vgg_type="vgg19",
        use_input_norm=True,
        range_norm=False,
        perceptual_weight=1.0,
        style_weight=0.0,
        criterion="l1",
    ):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm,
        )

        self.criterion_type = criterion
        if self.criterion_type == "l1":
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == "fro":
            self.criterion = None
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == "fro":
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p="fro") * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == "fro":
                    style_loss += (
                        torch.norm(self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p="fro")
                        * self.layer_weights[k]
                    )
                else:
                    style_loss += (
                        self.criterion(self._gram_mat(x_features[k]), self._gram_mat(gt_features[k]))
                        * self.layer_weights[k]
                    )
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


@LOSS_REGISTRY.register()
class GANFeatLoss(nn.Module):
    """Define feature matching loss for gans

    Args:
        criterion (str): Support 'l1', 'l2', 'charbonnier'.
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, criterion="l1", loss_weight=1.0, reduction="mean"):
        super(GANFeatLoss, self).__init__()
        if criterion == "l1":
            self.loss_op = L1Loss(loss_weight, reduction)
        elif criterion == "l2":
            self.loss_op = MSELoss(loss_weight, reduction)
        elif criterion == "charbonnier":
            self.loss_op = CharbonnierLoss(loss_weight, reduction)
        else:
            raise ValueError(f"Unsupported loss mode: {criterion}. Supported ones are: l1|l2|charbonnier")

        self.loss_weight = loss_weight

    def forward(self, pred_fake, pred_real):
        num_d = len(pred_fake)
        loss = 0
        for i in range(num_d):  # for each discriminator
            # last output is the final prediction, exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.loss_op(pred_fake[i][j], pred_real[i][j].detach())
                loss += unweighted_loss / num_d
        return loss * self.loss_weight


@LOSS_REGISTRY.register()
class SpectralLoss(nn.Module):
    """Spectral Fidelity loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(SpectralLoss, self).__init__()
        for param in self.parameters():
            param.requires_grad = False
        self.loss_weight = loss_weight

    def forward(self, sr_image, hr_image):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        # 计算HR和SR图像的傅里叶变换
        hr_fft = torch.fft.fft2(hr_image)
        sr_fft = torch.fft.fft2(sr_image)

        # 移动零频分量到中心
        hr_fft_shifted = torch.fft.fftshift(hr_fft)
        sr_fft_shifted = torch.fft.fftshift(sr_fft)

        # 计算频谱幅值并取对数
        hr_magnitude_spectrum = torch.log1p(torch.abs(hr_fft_shifted))
        sr_magnitude_spectrum = torch.log1p(torch.abs(sr_fft_shifted))

        # 标准化频谱幅值
        hr_magnitude_spectrum = (hr_magnitude_spectrum - hr_magnitude_spectrum.mean()) / hr_magnitude_spectrum.std()
        sr_magnitude_spectrum = (sr_magnitude_spectrum - sr_magnitude_spectrum.mean()) / sr_magnitude_spectrum.std()

        # 计算频谱损失
        loss = F.mse_loss(hr_magnitude_spectrum, sr_magnitude_spectrum)
        return loss * self.loss_weight


@LOSS_REGISTRY.register()
class SobelLoss(nn.Module):
    def __init__(self, loss_type="l1", loss_weight=1.0):
        super(SobelLoss, self).__init__()
        self.loss_weight = loss_weight
        assert loss_type in ["l1", "l2"], "loss_type must be 'l1' or 'l2'"
        self.loss_type = loss_type

        # Sobel 필터 정의
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)

        # (1, 1, 3, 3)로 reshape → conv2d에 사용 가능
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, sr_image, hr_image):
        """
        pred, target: (B, C, H, W) RGB 또는 grayscale tensor
        """
        B, C, H, W = sr_image.shape
        loss = 0.0

        for c in range(C):  # 채널 별로 독립적으로 gradient 계산
            pred_c = sr_image[:, c : c + 1, :, :]
            target_c = hr_image[:, c : c + 1, :, :]

            gx_pred = F.conv2d(pred_c, self.sobel_x, padding=1)
            gy_pred = F.conv2d(pred_c, self.sobel_y, padding=1)
            gx_target = F.conv2d(target_c, self.sobel_x, padding=1)
            gy_target = F.conv2d(target_c, self.sobel_y, padding=1)

            if self.loss_type == "l1":
                loss += F.l1_loss(gx_pred, gx_target) + F.l1_loss(gy_pred, gy_target)
            else:
                loss += F.mse_loss(gx_pred, gx_target) + F.mse_loss(gy_pred, gy_target)
        return (loss / C) * self.loss_weight

@LOSS_REGISTRY.register()
class GradientVariance(nn.Module):
    """Class for calculating GV loss between to RGB images
       :parameter
       patch_size : int, scalar, size of the patches extracted from the gt and predicted images
       cpu : bool,  whether to run calculation on cpu or gpu
        """
    def __init__(self, loss_weight=1.0, patch_size = 8, cpu=False):
        super(GradientVariance, self).__init__()
        self.loss_weight = loss_weight
        self.patch_size = patch_size
        # Sobel kernel for the gradient map calculation
        self.kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        self.kernel_y = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0)
        if not cpu:
            self.kernel_x = self.kernel_x.cuda()
            self.kernel_y = self.kernel_y.cuda()
        # operation for unfolding image into non overlapping patches
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size)

    def forward(self, output, target):
        # converting RGB image to grayscale
        gray_output = 0.2989 * output[:, 0:1, :, :] + 0.5870 * output[:, 1:2, :, :] + 0.1140 * output[:, 2:, :, :]
        gray_target = 0.2989 * target[:, 0:1, :, :] + 0.5870 * target[:, 1:2, :, :] + 0.1140 * target[:, 2:, :, :]

        # calculation of the gradient maps of x and y directions
        gx_target = F.conv2d(gray_target, self.kernel_x, stride=1, padding=1)
        gy_target = F.conv2d(gray_target, self.kernel_y, stride=1, padding=1)
        gx_output = F.conv2d(gray_output, self.kernel_x, stride=1, padding=1)
        gy_output = F.conv2d(gray_output, self.kernel_y, stride=1, padding=1)

        # unfolding image to patches
        gx_target_patches = self.unfold(gx_target)
        gy_target_patches = self.unfold(gy_target)
        gx_output_patches = self.unfold(gx_output)
        gy_output_patches = self.unfold(gy_output)

        # calculation of variance of each patch
        var_target_x = torch.var(gx_target_patches, dim=1)
        var_output_x = torch.var(gx_output_patches, dim=1)
        var_target_y = torch.var(gy_target_patches, dim=1)
        var_output_y = torch.var(gy_output_patches, dim=1)

        # loss function as a MSE between variances of patches extracted from gradient maps
        gradvar_loss = F.mse_loss(var_target_x, var_output_x) + F.mse_loss(var_target_y, var_output_y)

        return gradvar_loss * self.loss_weight