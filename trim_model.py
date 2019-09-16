import os
import torch
import argparse
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
import torch.nn as nn
import torch

num_inputs = 1024
dim_reduced = 256
num_classes = 2
num_bbox_reg_classes = 2


def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r


parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    #default="~/.torch/models/_detectron_35858933_12_2017_baselines_e2e_mask_rcnn_R-50-FPN_1x.yaml.01_48_14.DzEQe4wC_output_train_coco_2014_train%3Acoco_2014_valminusminival_generalized_rcnn_model_final.pkl",
    default="./pretrained_model/e2e_mask_rcnn_X_101_32x8d_FPN_1x.pth",
    help="path to detectron pretrained weight(.pkl)",
    type=str,
)
parser.add_argument(
    "--save_path",
    default="./pretrained_model/e2e_mask_rcnn_X_101_32x8d_FPN_1x_dp.pth",
    help="path to save the converted model",
    type=str,
)
parser.add_argument(
    "--cfg",
    default="configs/e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml",
    help="path to config file",
    type=str,
)

args = parser.parse_args()
#
DETECTRON_PATH = os.path.expanduser(args.pretrained_path)
print('detectron path: {}'.format(DETECTRON_PATH))

cfg.merge_from_file(args.cfg)
_d = torch.load(DETECTRON_PATH, map_location=torch.device('cpu'))
newdict = _d


cls_score = nn.Linear(num_inputs, num_classes)
bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)
mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
conv5_mask = nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)


newdict['model']['module.roi_heads.box.predictor.cls_score.weight'] = cls_score.weight
newdict['model']['module.roi_heads.box.predictor.cls_score.bias']   = cls_score.bias
newdict['model']['module.roi_heads.box.predictor.bbox_pred.weight'] = bbox_pred.weight
newdict['model']['module.roi_heads.box.predictor.bbox_pred.bias']   = bbox_pred.bias
newdict['model']['module.roi_heads.mask.predictor.mask_fcn_logits.weight'] = mask_fcn_logits.weight
newdict['model']['module.roi_heads.mask.predictor.mask_fcn_logits.bias'] = mask_fcn_logits.bias
newdict['model']['module.roi_heads.mask.predictor.conv5_mask.weight'] = conv5_mask.weight
newdict['model']['module.roi_heads.mask.predictor.conv5_mask.bias'] = conv5_mask.bias



torch.save(newdict, args.save_path)
print('saved to {}.'.format(args.save_path))