import cv2
import os
import numpy as np
import json
import imutils

def polypoints(mask):
    mask_list = []
    cnts = cv2.findContours(mask.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    # cnt = cnts[0]
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.005 * peri, True)
        approx = approx.reshape(-1).tolist()
        mask_list.append([approx])
    return mask_list