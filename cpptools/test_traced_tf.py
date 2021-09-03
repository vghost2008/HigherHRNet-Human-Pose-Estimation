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
import tensorflow as tf

from tf_group import TFGroup
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
def make_placeholder():
    data = [[1,17,30,1], [1,17,30,2], [1,17,30],[1,512,512,17],[1,512,512,17,1]]
    res = []
    for x in data:
        res.append(tf.placeholder(dtype=tf.float32,shape=x))
    return res

def main():
    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"weights/traced.pt")
    model = torch.jit.load(save_path)
    args = parse_args()
    check_config(cfg)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    ptag_k,pind_k,pval_k,pdet,ptags = make_placeholder()

    parser = TFGroup()
    r_keypoints = parser.inference(ptag_k,pind_k,pval_k,pdet,ptags)[0]
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    images_files = glob.glob(os.path.join(args.eval_folder,"*.jpg"))
    images_files += glob.glob(os.path.join(args.eval_folder,"*.png"))
    images_files += glob.glob(os.path.join(args.eval_folder,"*.jpeg"))
    sess = None

    pbar = tqdm(total=len(images_files))
    device = torch.device("cuda")
    for i, image_f in enumerate(images_files):
        image = cv2.imread(image_f,cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)[:,:,::-1]
        image = cv2.resize(image,(512,512))
        fimage = image.astype(np.float32)
        #if image.shape[0]<256:
            #image = cv2.resize(image,(image.shape[1]*2,image.shape[0]*2))
        image_name = os.path.basename(image_f)
        fimage = torch.Tensor(fimage).to(device)
        tag_k,ind_k,val_k,det,tags = model(fimage)
        tag_k,ind_k,val_k,det,tags =tag_k.cpu().detach().numpy(),ind_k.cpu().detach().numpy(),val_k.cpu().detach().numpy(),det.cpu().detach().numpy(),tags.cpu().detach().numpy()
        # size at scale 1.0
        ind_k = ind_k.astype(np.float32)
        det = det.transpose(0,3,2,1)
        tags = tags.transpose(0,3,2,1,4)
        #final_results nx17x5  5(x,y,scores,tags0,tags1,...)
        if sess is None:
            sess = tf.Session(config=tf_config)
        final_results = sess.run(r_keypoints,feed_dict={ptag_k:tag_k,pind_k:ind_k,pval_k:val_k,pdet:det,ptags:tags})

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
