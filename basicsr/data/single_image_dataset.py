import os
from os import path as osp
import pandas as pd
import cv2
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SingleTestImageDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(SingleTestImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None
        self.lq_folder = opt["dataroot_gt"] if opt["dataroot_gt"] is not None else opt["dataroot_lq"]
        self.scale = opt["scale"]

        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = [self.lq_folder]
            self.io_backend_opt["client_keys"] = ["lq"]
            self.paths = paths_from_lmdb(self.lq_folder)
        elif self.io_backend_opt["type"] == "csv":
            self.paths = []
            for root, _, fnames in sorted(os.walk(self.lq_folder)):
                for fname in fnames:
                    if fname.startswith("test") and fname.endswith(".csv"):
                        csv = pd.read_csv(osp.join(root, fname))
                        self.paths = csv.sample(n=10, random_state=1)["image_path"].tolist()

        elif "meta_info_file" in self.opt:
            with open(self.opt["meta_info_file"], "r") as fin:
                self.paths = [osp.join(self.lq_folder, line.split(" ")[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop("type"), **self.io_backend_opt)

        # load lq image
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, "lq")

        if self.opt["dataroot_gt"] is not None:
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gt_size = (img_gt.shape[1] // self.scale * self.scale, img_gt.shape[0] // self.scale * self.scale)

            uimg_gt = cv2.UMat(img_gt)
            uimg_gt = cv2.resize(uimg_gt, img_gt_size, interpolation=cv2.INTER_CUBIC)
            img_gt = uimg_gt.get()

            img_lq_size = (img_gt.shape[1] // self.scale, img_gt.shape[0] // self.scale)
            uimg_lq = cv2.resize(uimg_gt, img_lq_size, interpolation=cv2.INTER_CUBIC)
            img_lq = uimg_lq.get()

            # TODO: color space transform
            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
            img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
            # normalize
            if self.mean is not None or self.std is not None:
                normalize(img_lq, self.mean, self.std, inplace=True)
            return {"lq": img_lq, "lq_path": lq_path, "gt": img_gt}
        else:
            img_lq = imfrombytes(img_bytes, float32=True)
            img_gt_size = (img_lq.shape[1] * self.scale, img_lq.shape[0] * self.scale)
            uimg_lq = cv2.UMat(img_lq)
            uimg_gt = cv2.resize(uimg_lq, img_gt_size, interpolation=cv2.INTER_CUBIC)
            img_gt = uimg_gt.get()
            img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
            img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)

            # normalize
            if self.mean is not None or self.std is not None:
                normalize(img_lq, self.mean, self.std, inplace=True)
            return {"lq": img_lq, "lq_path": lq_path, "gt": img_gt}

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class SingleTrainImageDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(SingleTrainImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None
        self.gt_folder = opt["dataroot_gt"] if opt["dataroot_gt"] is not None else opt["dataroot_lq"]
        self.scale = opt["scale"]

        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = [self.gt_folder]
            self.io_backend_opt["client_keys"] = ["lq"]
            self.paths = paths_from_lmdb(self.gt_folder)
        elif self.io_backend_opt["type"] == "csv":
            self.paths = []
            for root, _, fnames in sorted(os.walk(self.gt_folder)):
                for fname in fnames:
                    if fname.startswith("train") and fname.endswith(".csv"):
                        csv = pd.read_csv(osp.join(root, fname))
                        self.paths = csv["image_path"].tolist()

        elif "meta_info_file" in self.opt:
            with open(self.opt["meta_info_file"], "r") as fin:
                self.paths = [osp.join(self.gt_folder, line.split(" ")[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop("type"), **self.io_backend_opt)

        # load lq image
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, "lq")

        scale = self.opt["scale"]

        if self.opt["dataroot_gt"] is not None:
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gt_size = (img_gt.shape[1] // self.scale * self.scale, img_gt.shape[0] // self.scale * self.scale)

            img_gt = cv2.resize(img_gt, img_gt_size, interpolation=cv2.INTER_CUBIC)

            img_lq_size = (img_gt.shape[1] // self.scale, img_gt.shape[0] // self.scale)
            img_lq = cv2.resize(img_gt, img_lq_size, interpolation=cv2.INTER_CUBIC)

            if self.opt["phase"] == "train":
                gt_size = self.opt["gt_size"]
                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
                # flip, rotation
                img_gt, img_lq = augment([img_gt, img_lq], self.opt["use_hflip"], self.opt["use_rot"])

            # TODO: color space transform
            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
            img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)

            # normalize
            if self.mean is not None or self.std is not None:
                normalize(img_lq, self.mean, self.std, inplace=True)
            return {"lq": img_lq, "gt_path": gt_path, "gt": img_gt}
        else:
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
            # normalize
            if self.mean is not None or self.std is not None:
                normalize(img_lq, self.mean, self.std, inplace=True)
            return {"lq": img_lq, "lq_path": gt_path}

    def __len__(self):
        return len(self.paths)
