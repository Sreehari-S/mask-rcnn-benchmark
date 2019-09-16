import os
import torch
import argparse
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
import torch.nn as nn
import torch

num_inputs = 1024
dim_reduced = 256
num_classes = 12
num_bbox_reg_classes = 12


def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r


parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    default="~/.torch/models/_detectron_35858933_12_2017_baselines_e2e_mask_rcnn_R-50-FPN_1x.yaml.01_48_14.DzEQe4wC_output_train_coco_2014_train%3Acoco_2014_valminusminival_generalized_rcnn_model_final.pkl",
    #default="model_0074000.pth",
    help="path to detectron pretrained weight(.pkl)",
    type=str,
)
parser.add_argument(
    "--save_path",
    default="./pretrained_model/mask_rcnn_R-50-FPN_1x_detectron_box_trained.pth",
    help="path to save the converted model",
    type=str,
)
parser.add_argument(
    "--cfg",
    default="configs/e2e_mask_rcnn_R_50_FPN_1x.yaml",
    help="path to config file",
    type=str,
)

args = parser.parse_args()
#
DETECTRON_PATH = os.path.expanduser(args.pretrained_path)
print('detectron path: {}'.format(DETECTRON_PATH))

cfg.merge_from_file(args.cfg)
_d = load_c2_format(cfg, DETECTRON_PATH)
newdict = _d
import pdb 
pdb.set_trace()
newdict['model'] = removekey(_d['model'],
                             ['cls_score.bias', 'cls_score.weight', 'bbox_pred.bias', 'bbox_pred.weight'])

cls_score = nn.Linear(num_inputs, num_classes)
bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)
mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
conv5_mask = nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)

#nn.init.normal_(cls_score.weight, mean=0, std=0.01)
#nn.init.constant_(cls_score.bias, 0)

#nn.init.normal_(bbox_pred.weight, mean=0, std=0.001)
#nn.init.constant_(bbox_pred.bias, 0)

#nn.init.normal_(mask_fcn_logits.weight, mean=0, std=0.001)
#nn.init.constant_(mask_fcn_logits.bias, 0)

#nn.init.normal_(conv5_mask.weight, mean=0, std=0.001)
#nn.init.constant_(conv5_mask.bias, 0)

newdict['model']['cls_score.weight'] = cls_score.weight
newdict['model']['cls_score.bias']   = cls_score.bias
newdict['model']['bbox_pred.weight'] = bbox_pred.weight
newdict['model']['bbox_pred.bias']   = bbox_pred.bias
newdict['model']['mask_fcn_logits.weight'] = mask_fcn_logits.weight
newdict['model']['mask_fcn_logits.bias'] = mask_fcn_logits.bias
newdict['model']['conv5_mask.weight'] = conv5_mask.weight
newdict['model']['conv5_mask.bias'] = conv5_mask.bias

#import pdb
#pdb.set_trace()

torch.save(newdict, args.save_path)
print('saved to {}.'.format(args.save_path))