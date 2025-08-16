import os
import sys
import math

import numpy as np
import cv2
from skimage import draw

import Utils
import config as cf


def safe_find_contours(img, mode, method):
    """Wrapper for OpenCV 3.x vs 4.x compatibility"""
    contours_info = cv2.findContours(img, mode, method)
    if len(contours_info) == 3:  # OpenCV 3.x
        _, contours, hierarchy = contours_info
    else:  # OpenCV 4.x
        contours, hierarchy = contours_info
    return contours, hierarchy


def fit_floorplan(data):

    # Thresholding by 0.5
    _, data_thresh = cv2.threshold(data, 0.5, 1, 0)
    data_thresh = np.uint8(data_thresh)

    # Find contours
    data_cnt, data_heri = safe_find_contours(data_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    data_cnt.sort(key=lambda x: cv2.contourArea(x), reverse=True)

    # Largest bounding box
    sub_x, sub_y, w, h = cv2.boundingRect(data_cnt[0])
    data_sub = data_thresh[sub_y:sub_y+h, sub_x:sub_x+w]

    # Contours on cropped
    data_cnt, data_heri = safe_find_contours(data_sub, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    data_cnt.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    epsilon = 0.005 * cv2.arcLength(data_cnt[0], True)
    approx = cv2.approxPolyDP(data_cnt[0], epsilon, True)

    # Regression analysis on edges
    x_lst = [0, ]
    y_lst = [0, ]
    for i in range(len(approx)):
        p1 = approx[i][0]
        p2 = approx[(i+1) % len(approx)][0]

        if (p2[0]-p1[0]) == 0:
            slope = 10
        else:
            slope = abs((p2[1]-p1[1]) / (p2[0]-p1[0]))

        if slope <= 1:
            s = int((p1[1] + p2[1]) / 2)
            y_lst.append(s)
        elif slope > 1:
            s = int((p1[0] + p2[0]) / 2)
            x_lst.append(s)

    x_lst.append(data_sub.shape[1])
    y_lst.append(data_sub.shape[0])
    x_lst.sort()
    y_lst.sort()

    # Merge close points
    diag = math.sqrt(math.pow(data_sub.shape[1], 2) + math.pow(data_sub.shape[0], 2))

    def merge_near(lst):
        group = [[0, ]]
        for i in range(1, len(lst)):
            if lst[i] - np.mean(group[-1]) < diag * 0.05:
                group[-1].append(lst[i])
            else:
                group.append([lst[i], ])
        group = [int(np.mean(x)) for x in group]
        return group

    x_lst = merge_near(x_lst)
    y_lst = merge_near(y_lst)

    # Build floorplan grid
    ans = np.zeros((data_sub.shape[0], data_sub.shape[1]))
    for i in range(len(x_lst)-1):
        for j in range(len(y_lst)-1):
            sample = data_sub[y_lst[j]:y_lst[j+1], x_lst[i]:x_lst[i+1]]
            score = sample.mean()
            if score >= 0.5:
                ans[y_lst[j]:y_lst[j+1], x_lst[i]:x_lst[i+1]] = 1

    # Final contours
    pred = np.uint8(ans)
    pred_cnt, pred_heri = safe_find_contours(pred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pred_cnt.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    polygon = [(p[0][1], p[0][0]) for p in pred_cnt[0][::-1]]

    Y = np.array([p[0]+sub_y for p in polygon])
    X = np.array([p[1]+sub_x for p in polygon])
    fp_pts = np.concatenate((Y[np.newaxis, :], X[np.newaxis, :]), axis=0)

    # Draw floorplan
    fp_pred = np.zeros(data.shape)
    rr, cc = draw.polygon(fp_pts[0], fp_pts[1])
    rr = np.clip(rr, 0, data.shape[0]-1)
    cc = np.clip(cc, 0, data.shape[1]-1)
    fp_pred[rr, cc] = 1

    return fp_pts, fp_pred


def run(fp_prob, fc_prob_up, fc_prob_down, height_pred):

    # Normalize floor-ceiling map
    scale = cf.camera_h / (height_pred - cf.camera_h)
    fc_prob_down_r = Utils.resize_crop(fc_prob_down, scale, cf.fp_size)

    # Fuse maps
    fp_prob_fuse = fp_prob * 0.5 + fc_prob_up * 0.25 + fc_prob_down_r * 0.25

    # Run 2D floorplan fitting
    fp_pts, fp_pred = fit_floorplan(fp_prob_fuse)

    return fp_pts, fp_pred
