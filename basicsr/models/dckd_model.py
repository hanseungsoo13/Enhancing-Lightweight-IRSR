import torch
import torch.nn.functional as F

from basicsr.archs import build_network
from basicsr.data.degradations import random_add_gaussian_noise_pt
from basicsr.losses import build_loss
from .sr_model import SRModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY

from ptflops import get_model_complexity_info
import random
import numpy as np
import os.path as osp
from collections import OrderedDict
from kornia.filters import gaussian_blur2d
from tqdm import tqdm


def noising(imgs, idx, img_range=1.0, revert=False):
    # imgs: torch tensor B,C,H,W

    if revert:
        imgs[idx["t"]] = imgs[idx["t"]].clone()
        imgs[idx["v"]] = imgs[idx["v"]].flip(dims=(-1,))
        imgs[idx["h"]] = imgs[idx["h"]].flip(dims=(-2,))
        imgs[idx["i"]] = img_range - imgs[idx["i"]]

    else:
        imgs[idx["i"]] = img_range - imgs[idx["i"]]
        imgs[idx["h"]] = imgs[idx["h"]].flip(dims=(-2,))
        imgs[idx["v"]] = imgs[idx["v"]].flip(dims=(-1,))
        imgs[idx["t"]] = imgs[idx["t"]].clone()

    return imgs


@MODEL_REGISTRY.register()
class DCKDModel(SRModel):
    def __init__(self, opt):
        super(DCKDModel, self).__init__(opt)

        # define teacher network
        if opt.get("network_t") is not None:
            self.net_t = build_network(opt["network_t"]).to(self.device)
            # self.print_network(self.net_t)
            self.net_t.eval()

            # load pretrained models
            load_path = opt["path"].get("pretrain_network_t")
            if load_path is not None:
                param_key = opt["path"].get("param_key_t", None)
                self.load_network(self.net_t, load_path, opt["path"].get("strict_load_t", True), param_key)

            for p in self.net_t.parameters():
                p.requires_grad = False
        else:
            self.net_t = None

        # define history network
        if opt.get("network_his") is not None:
            self.net_his = build_network(opt["network_g"]).to(self.device)
            self.net_his.eval()

            for p in self.net_his.parameters():
                p.requires_grad = False

            self.update_model_ema(0)
        else:
            self.net_his = None

        # define VQGAN network
        if opt.get("network_vqgan") is not None:
            self.net_vqgan = build_network(opt["network_vqgan"]).to(self.device)
            self.net_vqgan.eval()

            # load pretrained models
            load_path = opt["path"].get("pretrain_network_vqgan")
            if load_path is not None:
                param_key = opt["path"].get("param_key_vqgan", None)
                self.load_network(self.net_vqgan, load_path, opt["path"].get("strict_load_vqgan", True), param_key)

            for p in self.net_vqgan.parameters():
                p.requires_grad = False
        else:
            self.net_vqgan = None

        if opt.get("net_D") is not None:
            self.net_D = build_network(opt["net_D"]).to(self.device)
            self.net_D.eval()

            # load pretrained models
            load_path = opt["path"].get("pretrain_network_D")
            if load_path is not None:
                param_key = opt["path"].get("param_key_D", None)
                self.load_network(self.net_D, load_path, opt["path"].get("strict_load_D", True), param_key)

            for p in self.net_D.parameters():
                p.requires_grad = False

        # torch.save(self.net_vqgan.state_dict(), "experiments/pretrained_models/VQGAN/VQGAN_f16_n1024.pth")

    def init_training_settings(self):
        super().init_training_settings()

        # define losses
        train_opt = self.opt["train"]

        if train_opt.get("pixel_opt"):
            self.cri_pix = build_loss(train_opt["pixel_opt"]).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get("perceptual_opt"):
            self.cri_perceptual = build_loss(train_opt["perceptual_opt"]).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get("sobel_opt"):
            self.cri_edge = build_loss(train_opt["sobel_opt"]).to(self.device)
        else:
            self.cri_edge = None

        if train_opt.get("prompt_opt"):
            self.cri_prompt = build_loss(train_opt["prompt_opt"]).to(self.device)
            torch.cuda.empty_cache()
        else:
            self.cri_prompt = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError("Both pixel and perceptual losses are None.")

        if train_opt.get("spectral_opt"):
            self.cri_spectral = build_loss(train_opt["spectral_opt"]).to(self.device)
        else:
            self.cri_spectral = None

        if train_opt.get("logits_opt") is not None:
            self.cri_logits = build_loss(train_opt["logits_opt"]).to(self.device)
        else:
            self.cri_logits = None

        if train_opt.get("lcr_opt") is not None:
            self.cri_lcr = build_loss(train_opt["lcr_opt"]).to(self.device)
            self.prob = train_opt["noisy"].get("prob", 0.5)
        else:
            self.cri_lcr = None

        if train_opt.get("cl_opt") is not None:
            self.cri_cl = build_loss(train_opt["cl_opt"]).to(self.device)
            self.num_neg = train_opt.get("num_neg", 4)
            self.update_decay = train_opt.get("update_decay", 0.1)
            self.step = train_opt.get("step", [])

            # degradation
            self.gaussian_blur_prob = train_opt.get("gaussian_blur_prob", 1.0)
            self.resize_prob = train_opt.get("resize_prob", 0)
            self.gaussian_noise_prob = train_opt.get("gaussian_noise_prob", 0)
            self.gray_noise_prob = train_opt.get("gray_noise_prob", 0)
        else:
            self.cri_cl = None

        if train_opt.get("ce_opt") is not None:
            self.cri_ce = build_loss(train_opt["ce_opt"]).to(self.device)
        else:
            self.cri_ce = None

        if train_opt.get("gan_opt") is not None:
            self.gan_loss = build_loss(train_opt["gan_opt"]).to(self.device)
        else:
            self.gan_loss = None

        if train_opt.get("gv_opt") is not None:
            self.cri_gv = build_loss(train_opt["gv_opt"]).to(self.device)
        else:
            self.cri_gv = None

    def update_model_ema(self, decay=0.1):
        net_g_ema_params = dict(self.net_g_ema.named_parameters())
        net_his_params = dict(self.net_his.named_parameters())

        for k in net_his_params.keys():
            net_his_params[k].data.mul_(decay).add_(net_g_ema_params[k].data, alpha=1 - decay)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        s_out = self.net_g(self.lq)

        if self.net_t is not None:
            if self.cri_lcr is not None:
                ops = ["i", "h", "v", "t"]  # invert color, horizontal, vertical flip, transpose
                t_noising_idx = {
                    op: torch.nonzero(
                        torch.Tensor(np.random.choice([0, 1], size=self.lq.shape[0], p=[1 - self.prob, self.prob]))
                    ).squeeze()
                    for op in ops
                }
                t_lq = noising(self.lq.clone(), t_noising_idx)
                t_out, t_out_lcr = torch.chunk(self.net_t(torch.cat([self.lq, t_lq], dim=0)), 2, dim=0)
                t_out_lcr = noising(t_out_lcr, t_noising_idx, revert=True)
            else:
                t_out = self.net_t(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if self.cri_pix is not None:
            l_pix = self.cri_pix(s_out, self.gt)
            l_total += l_pix
            loss_dict["l_pix"] = l_pix

        # perceptual loss
        if self.cri_perceptual is not None:
            l_percep, l_style = self.cri_perceptual(s_out, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict["l_percep"] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict["l_style"] = l_style

        # Sobel Gradient Loss
        if self.cri_edge is not None:
            l_edge = self.cri_edge(s_out, t_out)
            if l_edge is not None:
                l_total += l_edge
                loss_dict["l_edge"] = l_edge

        # prompt loss
        if self.cri_prompt is not None:
            l_prompt = self.cri_prompt(s_out, self.gt, [1.0, 1.0, 1.0, 1.0, 0.5])
            l_total += l_prompt
            loss_dict["l_prompt"] = l_prompt

        # log spectrum loss
        if self.cri_spectral is not None:
            l_spec = self.cri_spectral(s_out, t_out)
            l_total += l_spec
            loss_dict["l_spec"] = l_spec

        if self.cri_logits is not None:
            l_ts = self.cri_logits(s_out, t_out)
            l_total += l_ts
            loss_dict["l_ts"] = l_ts

        if self.cri_lcr is not None:
            l_lcr = self.cri_lcr(s_out, t_out_lcr)
            l_total += l_lcr
            loss_dict["l_lcr"] = l_lcr
        
        if self.cri_gv is not None:
            l_gv = self.cri_gv(s_out, self.gt)
            l_total += l_gv
            loss_dict["l_gv"] = l_gv

        if self.cri_cl is not None:
            pos_sample = [t_out]

            # degradation
            neg_sample = [self.lq]
            for _ in range(self.num_neg):
                llq = self.lq
                ori_h, ori_w = llq.size()[2:4]
                scale = 1

                if np.random.uniform() < self.gaussian_blur_prob:
                    kx = random.randint(1, 5) * 2 + 1  # [3, 11]
                    ky = random.randint(1, 5) * 2 + 1
                    sx = random.random() * 1.9 + 0.1  # [0.1, 2]
                    sy = random.random() * 1.9 + 0.1
                    llq = gaussian_blur2d(llq, (kx, ky), (sx, sy))

                if np.random.uniform() < self.resize_prob:
                    updown_type = random.choices(["up", "down"], [0.25, 0.75])[0]
                    if updown_type == "up":
                        scale = np.random.uniform(1, 1.5)
                    elif updown_type == "down":
                        scale = np.random.uniform(0.5, 1)
                    mode = random.choice(["area", "bilinear", "bicubic"])
                    llq = F.interpolate(
                        llq,
                        size=(int(ori_h * scale), int(ori_w * scale)),
                        mode=mode,
                        align_corners=None if mode == "area" else False,
                    )

                if np.random.uniform() < self.gaussian_noise_prob:
                    llq = random_add_gaussian_noise_pt(
                        llq, sigma_range=[1, 30], clip=True, rounds=False, gray_prob=self.gray_noise_prob
                    )

                if scale != 1:
                    mode = random.choice(["area", "bilinear", "bicubic"])
                    llq = F.interpolate(
                        llq, size=(ori_h, ori_w), mode=mode, align_corners=None if mode == "area" else False
                    )

                neg_sample.append(llq)

            neg_sample = torch.cat(neg_sample, dim=0)
            neg_sample = list(torch.chunk(self.net_his(neg_sample), self.num_neg + 1, dim=0))

            sample = [s_out] + pos_sample + neg_sample
            latents = self.net_vqgan.get_feas(sample)

            l_cl = self.cri_cl(latents)
            l_total += l_cl
            loss_dict["l_cl"] = l_cl

        if self.cri_ce is not None:
            s_d = self.net_vqgan.encode(s_out)
            t_d = self.net_vqgan.encode(self.gt)

            l_ce = self.cri_ce(s_d, t_d)
            l_total += l_ce
            loss_dict["l_ce"] = l_ce

        # GAN loss
        if self.gan_loss is not None:
            pred_g_fake = self.net_D(s_out.detach())
            input_ref = self.gt
            self.var_ref = input_ref.to(self.device)
            pred_d_real = self.net_D(self.var_ref).detach()
            l_gl = (
                self.gan_loss(pred_d_real - torch.mean(pred_g_fake), False)
                + self.gan_loss(pred_g_fake - torch.mean(pred_d_real), True)
            ) / 2
            l_total += l_gl
            loss_dict["l_gl"] = l_gl

        loss_dict["l_total"] = l_total

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        if self.net_his is not None:
            i = (current_iter - 1) // 100000
            if current_iter % self.step[i] == 0:
                self.update_model_ema(self.update_decay)
