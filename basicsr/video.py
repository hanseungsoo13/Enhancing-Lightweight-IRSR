import cv2
import os
import sys
import yaml
import numpy as np
from os import path as osp
from PIL import Image

import torch
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.archs import build_network
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options

def center_crop(image, crop_size):
    h, w = image.shape[:2]
    ch, cw = crop_size
    top = (h - ch) // 2
    left = (w - cw) // 2

    image = image[top:top+ch, left:left+cw]

    return image

def Image_preprocessing(
    input_img
):
    from basicsr.utils import img2tensor

    input_img = input_img.astype(np.float32) / 255.0

    # BGR to RGB, HWC to CHW, numpy to tensor
    img_lq = img2tensor(input_img, bgr2rgb=True, float32=True)

    return img_lq.unsqueeze(0)

def transform_image(input_img_tensor, model, device="cpu"):
    from basicsr.utils import tensor2img

    """
    Args:
        input_img_tensor: (C, H, W) torch.Tensor, normalized to [0, 1]
        model: torch model
        device: "cuda" or "cpu"
    Returns:
        output_img: numpy ndarray, dtype=uint8, shape=(H, W, C)
    """
    img_lq = input_img_tensor.to(device)

    with torch.no_grad():

        # Inference
        output = model(img_lq)
        output = output.detach().cpu()
        sr_img = tensor2img(output)

    return sr_img  # (H, W, C) np.uint8

def load_yml(opt):
    opt["is_train"] = False
    opt["dist"] = False
    return opt

def load_model(opt):
    net_g = build_network(opt["network_g"])
    # load pretrained models
    load_path = opt["path"].get("pretrain_network_g", None)
    if load_path is not None:
        strict = opt["path"].get("strict_load_g", True)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        try:
            load_net = load_net[opt["path"].get("param_key_g")]
        except:
            load_net = load_net
        net_g.load_state_dict(load_net, strict=strict)
    net_g.eval()
    return net_g

def video_pipeline(root_path):
    opt, _ = parse_options(root_path, is_train=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "experiments/best_models/psrgan_net_g_50000"

    sr_model = load_model(load_yml(opt))
    print('model loaded')

    # 입력 비디오 열기
    cap = cv2.VideoCapture(opt['video']['path'])
    if not cap.isOpened():
        print("[ERROR] 비디오 파일을 열 수 없습니다.")
        print( opt['video']['path'])
        return 
    else:
        print('video loaded')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(fps * 20)

    # Crop & SR 설정
    crop_size = (height // 2, width // 2)  # 중앙 반만 사용
    scale = 4
    sr_crop_size = (crop_size[0] * scale, crop_size[1] * scale)


    # 출력 비디오 설정
    out = cv2.VideoWriter('/home/sshan/project/experiment/DCKD/results/video/output_car_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (sr_crop_size[1], sr_crop_size[0]))
    out2 = cv2.VideoWriter('/home/sshan/project/experiment/DCKD/results/video/output_car_video2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (sr_crop_size[1], sr_crop_size[0]))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            print(f"[INFO] 프레임 {frame_idx} 도달. 종료.")
            break

        # 프레임 SR 처리
        sr_model.to(device)
        
        crop = center_crop(frame, crop_size)
        
        input_image = Image_preprocessing(crop)
        output_array = transform_image(input_image, sr_model, device=device)

        image = cv2.UMat(crop)
        uimg_sr = cv2.resize(image, (sr_crop_size[1],sr_crop_size[0]), interpolation=cv2.INTER_AREA)
        img_sr = uimg_sr.get()
        print(f"[INFO] 프레임 {frame_idx} 처리 중...")

        # 출력
        out.write(output_array)
        out2.write(img_sr)

        frame_idx += 1
    
    print(f"[DEBUG] sr_frame: type={type(output_array)}, shape={getattr(output_array, 'shape', None)}")
    print(f"[DEBUG] VideoWriter size: {(sr_crop_size[1], sr_crop_size[0])}")

    cap.release()
    out.release()
    out2.release()
    print("[INFO] 비디오 저장 완료")


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    video_pipeline(root_path)