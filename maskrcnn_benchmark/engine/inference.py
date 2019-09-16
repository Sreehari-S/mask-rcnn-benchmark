# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os
import cv2
import xml.etree.ElementTree as ET
import torch
from tqdm import tqdm
import numpy as np

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.data.transforms.build import build_transforms
from ..utils.comm import is_main_process
from ..utils.comm import all_gather
from ..utils.comm import synchronize
import scipy.misc as sm
from maskrcnn_benchmark.utils import cv2_util
import pandas as pd
from PIL import Image
import pdb

def compute_prediction(output):
    if output.has_field("mask"):
        masks = output.get_field("mask")
        masker = Masker(threshold=0.5, padding=1)
        masks = masker([masks], [output])[0]
        output.add_field("mask", masks)    
        return output

def select_top_predictions(predictions):
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > 0.5).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    score_vals, idx = scores.sort(0, descending=True)
    return score_vals.numpy(),predictions[idx]

def crop(im, height, width,stride=250):
    patchlist = []
    k = 0
    im = np.asarray(im)
    imgheight,imgwidth,_ = im.shape
    for i in range(0,imgheight,stride):
        rlist = []
        for j in range(0,imgwidth,stride):
            x = j+width
            y = i+height
            if x > imgwidth or y > imgheight:
                continue
            rlist.append( (Image.fromarray(im[i:y,j:x,:]),j,i) )
        patchlist.append(rlist)
    return patchlist

def compute_on_dataset(model,transforms,device):
    save_dir_res = "/home/uavws/Sreehari/mask-rcnn-benchmark/results"
    valid_csv = "/home/uavws/Sreehari/DigestPath/coordinate_data/train_test_points_fold_3"
    valid_df = pd.read_csv(os.path.join(valid_csv,'validation.csv'))
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    count = 0
    
    image_files = []
    for i in range(len(valid_df.Image_Path)):
        file_path = valid_df.Image_Path[i][6:]
        image_files.append(file_path)
    mask_files = []

    for i in range(len(valid_df.Mask_Path)):
        mask_dir = valid_df.Mask_Path[i]
        if valid_df.Mask_Path[i] !='0':
            mask_dir = mask_dir[6:]
        mask_files.append(mask_dir)
    img_count = 0
    for i in range(len(image_files)):
        image = Image.open(image_files[i])
        img_count+=1
        print("Image",img_count)
        bboxes = []
        if mask_files[i]!= '0':
            tree = ET.parse(mask_files[i])
            root = tree.getroot()
            for box_ in root.findall('object'):
                bndbox = box_.find('bndbox')
                box = []
                for vals in bndbox:
                    box.append(int(vals.text))
                bboxes.append(box)
            print("Number of boxes",len(bboxes))
        else:
            mask = None
            print("None")
            continue

        predicted_boxes = []
        flag = 0
        with torch.no_grad():
            plist = crop(image,512,512,256)
            for rlist in plist:
                for i in range(len(rlist)):
                    img,x,y = rlist[i]
                    img, _ = transforms(img,None)
                    img = img.unsqueeze(0).cuda()
                    output = model(img)
                    output = output[0].to(cpu_device) 
                    score_vals,boxes_ = select_top_predictions(output)
                    n_boxes = boxes_.bbox.shape[0]
                    count+=1
                    print("count",count)
                    print(n_boxes)
                    if n_boxes!=0:
                        flag=1
                        for box_ in boxes_.bbox:
                            x1,y1,x2,y2 = box_
                            x1+= x
                            x2+= x
                            y1+= y
                            y2+= y
                            predicted_boxes.append([x1,y1,x2,y2])
            if flag==1:
                image = np.array(image).astype(np.uint8)
                for box in bboxes:
                    image = cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),(0,0,255),1)
                for box in predicted_boxes:
                    image = cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),(0,255,0),1)
                cv2.imwrite(os.path.join(save_dir_res,(str(img_count)+'_.png')),image)
                    # label_boxes = boxes_.get_field("labels").numpy()
                    # for box in boxes_:
                    #     pdb.set_trace()
                    #     x1,y1,x2,y2 = box.bbox

    #     with torch.no_grad():
    #         output = model(images)
    #         output = [o.to(cpu_device) for o in output]
    #         images = [np.transpose(i.to(cpu_device),(1,2,0)) for i in images.tensors]
    #         for img_,boxes_,target_ in zip(images,output,targets):
    #             boxes_ = compute_prediction(boxes_)
    #             score_vals,boxes_ = select_top_predictions(boxes_)
    #             masks = boxes_.get_field("mask").numpy()
    #             n_boxes = boxes_.bbox.shape[0]
    #             img = np.array(img_).astype(np.uint8)
    #             label_boxes = boxes_.get_field("labels").numpy()
    #             assert(n_boxes==len(score_vals))
    #             for i in range(n_boxes):
    #                 thresh = masks[i][0, :, :, None]
    #                 box = np.array(boxes_.bbox[i]).reshape(-1,4)[0].astype(np.int)                 
    #                 img = cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),1)
    #                 contours, hierarchy = cv2_util.findContours(
    #                 thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #                 fill = cv2.fillPoly(img.get().copy(), contours, color=(0,0,255))
    #                 alpha = 0.5
    #                 img = cv2.addWeighted(fill, alpha, img, 1 - alpha,0)
    #                 font = cv2.FONT_HERSHEY_SIMPLEX
    #                 class_ = train_classes[label_boxes[i]-1]
    #                 img = cv2.putText(img,class_+"({})".format(round(float(score_vals[i]),2)),(box[0],box[1]-5), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    #             #for i in range(len(target_.bbox)):
    #                # target = np.array(target_.bbox[i]).reshape(-1,4)[0].astype(np.int)
    #                 #img = cv2.rectangle(img,(target[0],target[1]),(target[2],target[3]),(0,0,255),1)
    #             cv2.imwrite(os.path.join(r"D:\Sreehari\maskrcnn-benchmark\results",(str(count)+'_.png')),img)
    #             count +=1
    #             print(count)
    #     results_dict.update(
    #         {img_id: result for img_id, result in zip(image_ids, output)}
    #     )
    # return results_dict

def compute_on_dataset_fp(model,device):
    pass

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        cfg,
        mode,
        model,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        #torch.distributed.get_world_size()
        #if torch.distributed.is_initialized()
        #else 1 
        1
    )
    # logger = logging.getLogger("maskrcnn_benchmark.inference")
    # dataset = data_loader.dataset
    # logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()
    transforms = build_transforms(cfg,is_train=False)
    if mode == "fp":
        predictions = compute_on_dataset_fp(model, transforms, device)
    elif mode == "valid":
        predictions = compute_on_dataset(model, transforms, device)
    else:
        raise Exception("Invalid mode for inference!")
    # predictions = compute_on_dataset(model, data_loader, device)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # logger.info(
    #     "Total inference time: {} ({} s / img per device, on {} devices)".format(
    #         total_time_str, total_time * num_devices / len(dataset), num_devices
    #     )
    # )
    ############################################################################################################
    # predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    # if not is_main_process():
    #     return

    # if output_folder:
    #     torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    # extra_args = dict(
    #     box_only=box_only,
    #     iou_types=iou_types,
    #     expected_results=expected_results,
    #     expected_results_sigma_tol=expected_results_sigma_tol,
    # )

    #return evaluate(dataset=dataset,
    #                predictions=predictions,
    #                output_folder=output_folder,
    #                **extra_args)
