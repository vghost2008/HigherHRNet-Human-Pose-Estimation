# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import argparse
import os
import pprint
import glob
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm
import numpy as np
import img_utils as wmli
import _init_paths
import models

torch.ops.load_library("/home/wj/ai/work/higher_hrnet_cpp/torchop/build/libhrnet_op.so")

from config import cfg
from config import check_config
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.group import HeatmapParser
from dataset import make_test_dataloader
from fp16_utils.fp16util import network_to_half
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.vis import save_debug_images
from utils.vis import save_valid_image
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_predsv1
from utils.transforms import get_multi_scale_size

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml",
                        type=str)
    parser.add_argument('--eval-folder',
                        help='eval images floder',
                        default='images/',
                        type=str)
    args = parser.parse_args()

    return args


def main():
    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"weights/traced_cop.pt")
    model = torch.jit.load(save_path)
    print(model.code)
    args = parse_args()
    check_config(cfg)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    parser = HeatmapParser(cfg)
    images_files = glob.glob(os.path.join(args.eval_folder,"*.jpg"))
    images_files += glob.glob(os.path.join(args.eval_folder,"*.png"))
    images_files += glob.glob(os.path.join(args.eval_folder,"*.jpeg"))

    pbar = tqdm(total=len(images_files))
    device = torch.device("cuda")
    for i, image_f in enumerate(images_files):
        image = cv2.imread(image_f,cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)[:,:,::-1]
        image = wmli.resize_img(image,(512,512),True,align=64)
        fimage = image.astype(np.float32)
        #if image.shape[0]<256:
            #image = cv2.resize(image,(image.shape[1]*2,image.shape[0]*2))
        image_name = os.path.basename(image_f)
        fimage = torch.Tensor(fimage).to(device)
        ans = model(fimage)
        final_results = ans

        if cfg.TEST.LOG_PROGRESS:
            pbar.update()
        #wj
        if True:
            prefix = os.path.join(final_output_dir,image_name)
            logger.info('=> write {}'.format(prefix))
            save_valid_image(image, final_results, prefix, dataset="COCO",color=(0,255,0))
            # save_debug_images(cfg, image_resized, None, None, outputs, prefix)


    if cfg.TEST.LOG_PROGRESS:
        pbar.close()


if __name__ == '__main__':
    main()
