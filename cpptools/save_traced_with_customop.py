# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lib.core.group import match_by_tag
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
import torch.nn as nn
import _init_paths
import models
torch.ops.load_library("/home/wj/ai/work/higher_hrnet_cpp/torchop/build/libhrnet_op.so")
#torch.ops.load_library("/home/wj/ai/work/torchop/build/libwarp_perspective.so")
from config import cfg
from config import check_config
from config import update_config
from core.inference import get_multi_stage_outputsv2
from core.inference import aggregate_results
from core.group import HeatmapParser,OnlyTopK
from dataset import make_test_dataloader
from fp16_utils.fp16util import network_to_half
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.vis import save_debug_images
from utils.vis import save_valid_image
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
import wml_utils as wmlu

torch.multiprocessing.set_sharing_strategy('file_system')
#wmlu.show_dict(torch.ops.__dict__)
#print(torch.ops.hrnet_ops)
match_by_tag = torch.ops.hrnet_ops.match_by_tag
hrnet_refine = torch.ops.hrnet_ops.hrnet_refine


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args

@torch.jit.script
def get_tensor_size(x): 
    return torch.tensor([x.shape[-1],x.shape[-2]])

@torch.jit.script
def get_resize_size(x): 
    new_h = 512
    new_w = (x.shape[-1]+32)//64*64
    return torch.tensor([new_w,new_h]),torch.tensor([x.shape[-1],x.shape[-2]])
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


class CPPModel(nn.Module):
    def __init__(self,device=None):
        super().__init__()
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

        if device is not None:
            model = model.to(device)

        self.device = device
        self.model = model 
        self.parser = OnlyTopK(cfg)
        self.model.eval()
        self.base_size = None


    def forward(self,x):
        if self.device is not None:
            x = x.to(self.device)
        x = x/255.0
        mean = torch.tensor([0.485, 0.456, 0.406],dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225],dtype=torch.float32)
        mean = mean.view([3,1,1])
        std = std.view([3,1,1])
        if self.device is not None:
            mean = mean.to(self.device)
            std = std.to(self.device)
        x = x.permute(2,0,1)
        x = (x-mean)/std
    
        x = x.unsqueeze(0).cuda()

        new_size,old_size = get_resize_size(x)
        self.base_size = new_size
        image_resized = torch.nn.functional.interpolate(
                x,
                size=(512,512),
                mode='bilinear',
            )
    
        outputs, heatmaps, tags = get_multi_stage_outputsv2(
                        cfg, self.model, image_resized, cfg.TEST.FLIP_TEST,
                        cfg.TEST.PROJECT2IMAGE, self.base_size
                    )
        final_heatmaps, tags_list = aggregate_results(
                        cfg, 1, None, [], heatmaps, tags
                    )
    
        final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
        tags = torch.cat(tags_list, dim=4)
        tag_k,ind_k,val_k,det = self.parser.parse(
                    final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
                )
        tag_k,ind_k,val_k =tag_k.cpu().detach(),ind_k.cpu().detach(),val_k.cpu().detach() 
        tag_k,loc_k,val_k = torch.squeeze(tag_k,0),torch.squeeze(ind_k,0),torch.squeeze(val_k,0)

        loc_k = loc_k.to(torch.float32)
        ans = match_by_tag(tag_k,loc_k,val_k,0.1,1.0,True)
        ans[...,:2] = ans[...,:2]*old_size.to(torch.float32)/new_size.to(torch.float32)

        '''
        ans = torch.unsqueeze(ans,0)
        det = det.cpu().detach()
        tags= tags.cpu().detach()
        det = det.permute(0,2,3,1)
        tags = tags.permute(0,2,3,1,4)
        ans = hrnet_refine(ans,det,tags)
        ans = ans[0]
        '''

        #ans = match_by_tag(tag_k,loc_k,val_k,torch.tensor(0.1,dtype=torch.float64),torch.tensor(1.0,dtype=torch.float64),torch.tensor(True))
        #return shape [1,17,30,1], [1,17,30,2] (x,y) [1,17,30],[1,17,512,512],[1,17,512,512,1]
        return ans

def main():
    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"weights/traced_cop.pt")
    device = torch.device("cuda")
    model = CPPModel(device)
    input = torch.randn([512,512,3],dtype=torch.float32)
    input = input.to(device)
    traced_model = torch.jit.trace(model, input)
    print(traced_model.code)
    print(f"Save path {save_path}")
    traced_model.save(save_path)
    v = traced_model(input)
    print(v)



if __name__ == '__main__':
    main()