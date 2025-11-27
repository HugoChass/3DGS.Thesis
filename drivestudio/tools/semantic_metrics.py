from typing import Literal, Dict, List, Optional, Callable
from tqdm import tqdm, trange
import numpy as np
import os
import logging
import imageio

import torch
from torch import Tensor
from torch.nn import functional as F
from skimage.metrics import structural_similarity as ssim

from datasets.base import SplitWrapper
from models.trainers.base import BasicTrainer
from utils.visualization import (
    to8b,
    depth_visualizer,
)

logger = logging.getLogger()


def _ensure_list(x):
    """Allow both a single array or a list of arrays."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def _flatten_labels(preds, gts, ignore_label=None):
    """
    Flatten predicted and GT labels into 1D arrays, applying ignore mask if given.
    preds, gts: list of (H, W) numpy arrays or a single numpy array.
    """
    preds = _ensure_list(preds)
    gts = _ensure_list(gts)
    print(preds.shape, gts.shape)

    assert len(preds) == len(gts), "preds and gts must have same length"

    all_pred = []
    all_gt = []

    for p, g in zip(preds, gts):
        assert p.shape == g.shape, "pred and gt must have same shape per image"
        p_flat = p.reshape(-1)
        g_flat = g.reshape(-1)
        if ignore_label is not None:
            mask = (g_flat != ignore_label)
            p_flat = p_flat[mask]
            g_flat = g_flat[mask]
        all_pred.append(p_flat)
        all_gt.append(g_flat)

    pred_flat = np.concatenate(all_pred, axis=0)
    gt_flat = np.concatenate(all_gt, axis=0)
    return pred_flat, gt_flat

def compute_confusion_matrix(preds, gts, num_classes, ignore_label=None):
    """
    Compute confusion matrix of shape (num_classes, num_classes):
      rows   = ground truth class
      cols   = predicted class
    """
    pred_flat, gt_flat = _flatten_labels(preds, gts, ignore_label=ignore_label)

    # filter out anything outside [0, num_classes-1] just in case
    valid_mask = (gt_flat >= 0) & (gt_flat < num_classes) & \
                 (pred_flat >= 0) & (pred_flat < num_classes)
    gt_flat = gt_flat[valid_mask]
    pred_flat = pred_flat[valid_mask]

    confmat = np.bincount(
        num_classes * gt_flat.astype(np.int64) + pred_flat.astype(np.int64),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    return confmat

def compute_miou_from_confmat(confmat):
    """
    confmat: (C, C) array
    returns:
      per_class_iou: (C,) array
      miou: scalar
    """
    tp = np.diag(confmat).astype(np.float64)
    fp = confmat.sum(axis=0) - tp       # predicted as class c but GT != c
    fn = confmat.sum(axis=1) - tp       # GT class c but predicted != c
    denom = tp + fp + fn

    # Avoid division by zero
    per_class_iou = np.zeros_like(tp, dtype=np.float64)
    valid = denom > 0
    per_class_iou[valid] = tp[valid] / denom[valid]

    miou = per_class_iou[valid].mean() if np.any(valid) else 0.0
    return per_class_iou, miou

def compute_precision_recall_f1_from_confmat(confmat):
    """
    confmat: (C, C) array
    returns dict with per-class precision, recall and F1.
    """
    tp = np.diag(confmat).astype(np.float64)
    fp = confmat.sum(axis=0) - tp
    fn = confmat.sum(axis=1) - tp

    precision = np.zeros_like(tp)
    recall = np.zeros_like(tp)
    f1 = np.zeros_like(tp)

    # Precision: TP / (TP + FP)
    denom_p = tp + fp
    valid_p = denom_p > 0
    precision[valid_p] = tp[valid_p] / denom_p[valid_p]

    # Recall: TP / (TP + FN)
    denom_r = tp + fn
    valid_r = denom_r > 0
    recall[valid_r] = tp[valid_r] / denom_r[valid_r]

    # F1: 2 * P * R / (P + R)
    denom_f = precision + recall
    valid_f = denom_f > 0
    f1[valid_f] = 2.0 * precision[valid_f] * recall[valid_f] / denom_f[valid_f]

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def compute_semantic_metrics(preds, gts, num_classes, ignore_label=None):
    """
    High-level convenience wrapper:
      - preds/gts: list of (H, W) arrays (or single array)
      - num_classes: number of semantic classes (excluding ignore)
      - ignore_label: GT label to ignore (e.g., 255), can be None
    Returns:
      {
        "confmat": (C, C),
        "per_class_iou": (C,),
        "miou": float,
        "per_class_precision": (C,),
        "per_class_recall": (C,),
        "per_class_f1": (C,),
      }
    """
    print(preds.shape, gts.shape)
    confmat = compute_confusion_matrix(preds, gts, num_classes, ignore_label)
    per_class_iou, miou = compute_miou_from_confmat(confmat)
    prf = compute_precision_recall_f1_from_confmat(confmat)

    return {
        "confmat": confmat,
        "per_class_iou": per_class_iou,
        "miou": miou,
        "per_class_precision": prf["precision"],
        "per_class_recall": prf["recall"],
        "per_class_f1": prf["f1"],
    }