from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

import torch.ao.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping, get_default_qconfig_mapping

from .base_model import BaseModel
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class GANModel(BaseModel):
    """Train with pixel-VGG-GAN loss"""

    def __init__(self, opt):
        super(GANModel, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.net_g = build_network(opt["network_g"])
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt["path"].get("pretrain_network_g", None)
        if load_path is not None:
            param_key = self.opt["path"].get("param_key_g", None)
            self.load_network(self.net_g, load_path, self.opt["path"].get("strict_load_g", True), param_key)

        if self.is_train:
            self.init_training_settings()

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_training_settings(self):
        train_opt = self.opt["train"]  # training option
        self.net_g.train()  # set training mode,for BN

        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
            # load pretrained model
            load_path = self.opt["path"].get("pretrain_network_g", None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt["path"].get("strict_load_g", True), "params_ema")
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        if train_opt.get("pixel_opt"):
            self.cri_pix = build_loss(train_opt["pixel_opt"]).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get("perceptual_opt"):
            self.cri_perceptual = build_loss(train_opt["perceptual_opt"]).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError("Both pixel and perceptual losses are None.")

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f"Params {k} will not be optimized.")

        optim_type = train_opt["optim_g"].pop("type")
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt["optim_g"])
        self.optimizers.append(self.optimizer_g)

    # ----------------------------------------
    # load pre-trained G and D model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt["path"]["pretrained_network_g"]
        if load_path_G is not None:
            print("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.net_g)

    # ----------------------------------------
    # save model
    # ----------------------------------------

    def save(self, epoch, current_iter):
        if hasattr(self, "net_g_ema"):
            self.save_network([self.net_g, self.net_g_ema], "net_g", current_iter, param_key=["params", "params_ema"])
        else:
            self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(
            lr_scheduler.MultiStepLR(
                self.G_optimizer, self.opt_train["G_scheduler_milestones"], self.opt_train["G_scheduler_gamma"]
            )
        )

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict["l_pix"] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict["l_percep"] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict["l_style"] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    # ----------------------------------------
    # Quantization
    # ----------------------------------------
    def calibrate(self, model, data_loader):
        model.eval()
        with torch.no_grad():
            for idx, val_data in enumerate(data_loader):
                self.feed_data(val_data)
                model(self.lq)

    def quantization(self, data_loader):
        from torch.ao.quantization import QConfig, QConfigMapping, MinMaxObserver, default_observer
        from torch.ao.quantization.observer import PerChannelMinMaxObserver

        torch.backends.quantized.engine = "fbgemm"
        print("Starting quantization process...")
        # Specify how to quantize the model
        per_channel_obs = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)

        qconfig_global = QConfig(
            activation=default_observer.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
            weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
        )

        qconfig_depthwise = QConfig(
            activation=default_observer.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
            weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
        )

        qconfig_mapping = QConfigMapping().set_global(qconfig_global)

        depthwise_modules = [
            "KD.c1_d.0",
            "KD.c1_r.0",
            "KD.c2_d.0",
            "KD.c3_d.0",
            "KD.c4.0",
            "KD.c5.0",
            "KD.esa.conv1.0",
            "KD.esa.conv2.0",
            "body.sub.0.res.0",
            "body.sub.0.res.3",
            "body.sub.1.res.0",
            "body.sub.1.res.3",
        ]
        for name in depthwise_modules:
            qconfig_mapping.set_module_name(name, qconfig_depthwise)

        example_inputs = next(iter(data_loader))  # get an example input
        prepared_model = prepare_fx(self.net_g, qconfig_mapping, example_inputs)  # fuse modules and insert observers
        self.calibrate(prepared_model, data_loader)  # run calibration on sample data
        prepared_model = prepared_model.to("cpu")
        self.device = "cpu"
        quantized_model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model
        return quantized_model

    # ----------------------------------------
    # test and inference
    # ----------------------------------------
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt["rank"] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"].get("metrics") is not None
        use_pbar = self.opt["val"].get("pbar", False)
        use_img = self.opt["val"].get("use_img", True)
        rescale = self.opt["val"].get("rescale", False)

        if with_metrics:
            if not hasattr(self, "metric_results"):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt["val"]["metrics"].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit="image")

        if with_metrics:
            all_metrics = []  # 모든 이미지의 metric 결과를 저장할 리스트

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            if rescale:  # [-1, 1] -> [0, 1]
                visuals["result"] = torch.clamp_(visuals["result"], -1, 1) * 0.5 + 0.5
                visuals["gt"] = visuals["gt"] * 0.5 + 0.5

            sr_img = tensor2img([visuals["result"]])
            if use_img:
                metric_data["img"] = sr_img
            else:
                metric_data["img"] = visuals["result"]
            if "gt" in visuals:
                if use_img:
                    gt_img = tensor2img([visuals["gt"]])
                    metric_data["img2"] = gt_img
                else:
                    metric_data["img2"] = visuals["gt"]
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt["is_train"]:
                    save_img_path = osp.join(
                        self.opt["path"]["visualization"], img_name, f"{img_name}_{current_iter}.png"
                    )
                else:
                    if self.opt["val"]["suffix"]:
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png',
                        )
                    else:
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"], dataset_name, f'{img_name}_{self.opt["name"]}.png'
                        )
                imwrite(sr_img, save_img_path)

            if with_metrics:
                metric_results_per_image = {}
                for name, opt_ in self.opt["val"]["metrics"].items():
                    result = calculate_metric(metric_data, opt_)
                    metric_results_per_image[name] = result
                    self.metric_results[name] += result
                all_metrics.append((img_name, metric_results_per_image))
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f"Test {img_name}")
        if use_pbar:
            pbar.close()

        if with_metrics:
            # 모든 이미지 metric 결과를 하나의 파일로 저장
            metrics_save_path = osp.join(self.opt["path"]["visualization"], f"{dataset_name}_all_metrics.txt")
            with open(metrics_save_path, 'w') as f:
                for img_name, metrics in all_metrics:
                    metric_str = '  '.join([f"{k}: {v}" for k, v in metrics.items()])
                    f.write(f"{img_name}  {metric_str}\n")

            for metric in self.metric_results.keys():
                self.metric_results[metric] /= idx + 1
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
            if hasattr(self, "best_metric_results"):
                log_str += (
                    f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                    f'{self.best_metric_results[dataset_name][metric]["iter"]} iter'
                )
            log_str += "\n"

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{dataset_name}/{metric}", value, current_iter)

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H images
    # ----------------------------------------
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG, netD and netF
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.net_g)
        print(msg)
        if self.is_train:
            msg = self.describe_network(self.netD)
            print(msg)
            if self.opt_train["F_lossfn_weight"] > 0:
                msg = self.describe_network(self.netF)
                print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.net_g)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.net_g)
        if self.is_train:
            msg += self.describe_network(self.netD)
            if self.opt_train["F_lossfn_weight"] > 0:
                msg += self.describe_network(self.netF)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.net_g)
        return msg
