import logging
import torch
from os import path as osp
import io

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.metrics import calculate_metric
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs, tensor2img
from basicsr.utils import imwrite
from basicsr.utils.options import dict2str, parse_options

import torch.quantization


def quantization_pipeline(root_path, save=False):

    opt, _ = parse_options(root_path, is_train=False)

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt["path"]["log"], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name="basicsr", log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create model
    model = build_model(opt)

    # create test dataset and dataloader
    dataset_opt = opt["datasets"]["quantize_dataset"]
    dataset_opt["phase"] = "test"
    dataset_opt["scale"] = opt["scale"]
    test_set = build_dataset(dataset_opt)
    test_loader = build_dataloader(
        test_set, dataset_opt, num_gpu=opt["num_gpu"], dist=opt["dist"], sampler=None, seed=opt["manual_seed"]
    )

    # Quantization
    qt_model = model.quantization(test_loader)

    print("Quantization complete.")

    buffer = io.BytesIO()
    torch.save(qt_model.state_dict(), buffer)
    size_mb = buffer.getbuffer().nbytes / (1024 * 1024)
    print(f"✅ Quantized model file size (state_dict): {size_mb:.2f} MB")

    if save:
        torch.jit.save(
            torch.jit.script(qt_model),
            "/home/sshan/project/experiment/DCKD/experiments/best_models/jit_quantized_model.pt",
        )

    return qt_model


def get_current_visuals(lq, output, gt):
    from collections import OrderedDict

    out_dict = OrderedDict()
    out_dict["lq"] = lq.detach().cpu()
    out_dict["result"] = output.detach().cpu()
    out_dict["gt"] = gt.detach().cpu()
    return out_dict


def test_model(model, root_path):
    opt, _ = parse_options(root_path, is_train=False)
    test_loaders = []
    use_img = opt["val"].get("use_img", True)
    with_metrics = opt["val"].get("metrics") is not None
    if with_metrics:
        metric_results = {metric: 0 for metric in opt["val"]["metrics"].keys()}

    for dataset_name, dataset_opt in sorted(opt["datasets"].items()):
        if dataset_name.startswith("test"):
            test_set = build_dataset(dataset_opt)
            test_loader = build_dataloader(
                test_set, dataset_opt, num_gpu=opt["num_gpu"], dist=opt["dist"], sampler=None, seed=opt["manual_seed"]
            )
            test_loaders.append(test_loader)

    for test_loader in test_loaders:
        model.eval()
        model.to("cpu")
        metric_data = dict()
        test_set_name = test_loader.dataset.opt["name"]
        print(f"Testing {test_set_name}...")
        with torch.no_grad():
            for idx, val_data in enumerate(test_loader):
                lq = val_data["lq"].to("cpu")
                gt = val_data["gt"].to("cpu")

                output = model(lq)
                visuals = get_current_visuals(lq, output, gt)

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
                    del gt

                # tentative for out of GPU memory
                del lq
                del output

                if with_metrics:
                    # calculate metrics
                    for name, opt_ in opt["val"]["metrics"].items():
                        metric_results[name] += calculate_metric(metric_data, opt_)
            if with_metrics:
                for metric in metric_results.keys():
                    metric_results[metric] /= idx + 1
                    print(f"Metric {metric}: {metric_results[metric]:.4f}")


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    qt_model = quantization_pipeline(root_path, save=True)
    test_model(qt_model, root_path)
