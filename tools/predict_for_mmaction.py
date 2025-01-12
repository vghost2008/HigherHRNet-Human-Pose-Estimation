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
import sys
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
import pickle
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


import _init_paths
import models

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
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml',
                        type=str)
    parser.add_argument('--eval-folder',
                        help='eval images floder',
                        default='/home/wj/ai/mldata/Le2i/FallDown/test/images/1',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args

# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )

def main():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)
    cfg.defrost()
    cfg.TEST.MODEL_FILE = "models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth"
    cfg.freeze()

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    dump_input = torch.rand(
        (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
    )
    logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth.tar'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    if cfg.MODEL.NAME == 'pose_hourglass':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

    parser = HeatmapParser(cfg)

    for dir in os.listdir(args.eval_folder):
        cur_dir = os.path.join(args.eval_folder,dir)
        if not os.path.isdir(cur_dir):
            continue
        print(f"Process dir {dir}.")
        if cur_dir[-1]=="/":
            cur_dir = cur_dir[:-1]
        all_imgs = glob.glob(os.path.join(cur_dir,"*.jpg"))
        if len(all_imgs) == 0:
            print(f"ERROR: Find images in {cur_dir} faild.")
        total_nr = len(all_imgs)
        all_imgs = []
        for i in range(total_nr):
            all_imgs.append(os.path.join(cur_dir,f"img_{i+1:05d}.jpg"))

        save_path = cur_dir+".pkl"
        cur_kp_result = []
        cur_scores_result = []
        img_shape = None
        for i,ifn in enumerate(all_imgs):
            sys.stdout.write(f"\r{i+1}/{total_nr}.   ")
            image = cv2.imread(ifn, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)[:, :, ::-1]
            img_shape = image.shape[:2]
            # size at scale 1.0
            base_size, center, scale = get_multi_scale_size(
                image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
            )

            with torch.no_grad():
                final_heatmaps = None
                tags_list = []
                for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
                    input_size = cfg.DATASET.INPUT_SIZE
                    image_resized, center, scale = resize_align_multi_scale(
                        image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                    )
                    image_resized = transforms(image_resized)
                    image_resized = image_resized.unsqueeze(0).cuda()

                    outputs, heatmaps, tags = get_multi_stage_outputs(
                        cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                        cfg.TEST.PROJECT2IMAGE, base_size
                    )

                    final_heatmaps, tags_list = aggregate_results(
                        cfg, s, final_heatmaps, tags_list, heatmaps, tags
                    )

                final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
                tags = torch.cat(tags_list, dim=4)
                grouped, scores = parser.parse(
                    final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
                )
                #final_results nx17x5  5(x,y,scores,tags0,tags1,...)
                final_results = get_final_preds(
                    grouped, center, scale,
                    [final_heatmaps.size(3), final_heatmaps.size(2)]
                )
                final_results = np.array(final_results)
                if len(final_results)>0:
                    cur_kp_result.append(final_results[:,:,:2])
                    cur_scores_result.append(final_results[:,:,2])
                else:
                    cur_kp_result.append(final_results)
                    cur_scores_result.append(final_results)

        save_data = {
            'frame_dir':dir,
            'img_shape':img_shape,
            'original_shape':img_shape,
            'total_frames':total_nr,
            'keypoint':cur_kp_result,
            'keypoint_score':cur_scores_result,
        }
        with open(save_path,"wb") as f:
            pickle.dump(save_data,f)

if __name__ == '__main__':
    main()
