#!/usr/env/bin python3.6

import io
import re
import random
from operator import itemgetter
from pathlib import Path
from itertools import repeat
from functools import partial
from typing import Any, Callable, BinaryIO, Dict, List, Match, Pattern, Tuple, Union

from bounds import ConstantBounds,TagBounds, PreciseBounds
import torch
import numpy as np
from torch import Tensor
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from MySampler import Sampler
import os
from utils import id_, map_, class2one_hot, resize_im, soft_size
from utils import simplex, sset, one_hot, dice_batch
from argparse import Namespace
import os
import pandas as pd

import imageio

def dice3dn(all_grp,all_inter_card,all_card_gt,all_card_pred, metric_axis,pprint=False):
    #print(all_card_gt.shape)
    _,C = all_card_gt.shape
    unique_patients = torch.unique(all_grp)
    #print(sum(unique_patients == 0))
    unique_patients = unique_patients[unique_patients != torch.ones_like(unique_patients)*666]
    #unique_patients = unique_patients[unique_patients != 666]
    #print(unique_patients)
    batch_dice = torch.zeros((len(unique_patients), C))
    for i, p in enumerate(unique_patients):
        inter_card_p = torch.einsum("bc->c", [torch.masked_select(all_inter_card, all_grp == p).reshape((-1, C))])
        card_gt_p= torch.einsum("bc->c", [torch.masked_select(all_card_gt, all_grp == p).reshape((-1, C))])
        card_pred_p= torch.einsum("bc->c", [torch.masked_select(all_card_pred, all_grp == p).reshape((-1, C))])
        #if p == 0:
        #    print("inter_card_p:",inter_card_p.detach())
        #    print("card_gt_p:", card_gt_p.detach())
        #    print("card_pred_p:",card_pred_p.detach())
        #print(card_gt_p.shape)
        dice_3d = (2 * inter_card_p + 1e-8) / ((card_pred_p + card_gt_p)+ 1e-8)
        if pprint:
            dice_3d = torch.round(dice_3d * 10**2) / (10**2)
            print(p,dice_3d)
        batch_dice[i,...] = dice_3d

    indices = torch.tensor(metric_axis)
    dice_3d = torch.index_select(batch_dice, 1, indices)
    dice_3d_mean = dice_3d.mean(dim=0)
    print('metric_axis dice',dice_3d_mean)
    dice_3d_sd = dice_3d.std(dim=0)
    [dice_3d, dice_3d_sd] = map_(lambda t: t.mean(), [dice_3d_mean, dice_3d_sd])

    return dice_3d.item(), dice_3d_sd.item()


def run_dices(args: Namespace) -> None:

    for folder in args.folders:
        subfolders = args.subfolders
        all_dices=[0] * len(subfolders)
        for i, subfolder in enumerate(subfolders):
            print(subfolder)
            epc = int(subfolder.split('r')[1])
            dice_i = dice3d(args.base_folder, folder, subfolder, args.grp_regex, args.gt_folder)
            all_dices[epc] = dice_i

        df = pd.DataFrame({"3d_dice": all_dices})
        df.to_csv(Path(args.save_folder, 'dice_3d.csv'), float_format="%.4f", index_label="epoch")


def dice3d(base_folder, folder, subfoldername, grp_regex, gt_folder, C):
    if base_folder == '':
        work_folder = Path(folder, subfoldername)
    else:
        work_folder = Path(base_folder,folder, subfoldername)
    #print(work_folder)
    filenames = map_(lambda p: str(p.name), work_folder.glob("*.png"))
    grouping_regex: Pattern = re.compile(grp_regex)

    stems: List[str] = [Path(filename).stem for filename in filenames]  # avoid matching the extension
    matches: List[Match] = map_(grouping_regex.match, stems)
    patients: List[str] = [match.group(0) for match in matches]

    unique_patients: List[str] = list(set(patients))
    #print(unique_patients)
    batch_dice = torch.zeros((len(unique_patients), C))
    for i, patient in enumerate(unique_patients):
        patient_slices = [f for f in stems if f.startswith(patient)]
        w,h = [256,256]
        n = len(patient_slices)
        t_seg = np.ndarray(shape=(w, h, n))
        t_gt = np.ndarray(shape=(w, h, n))
        for slice in patient_slices:
            slice_nb = int(re.split(grp_regex, slice)[1])
            seg = imageio.imread(str(work_folder)+'/'+slice+'.png')
            gt = imageio.imread(str(gt_folder )+'/'+ slice+'.png')
            if seg.shape != (w, h):
                seg = resize_im(seg, 36)
            if gt.shape != (w, h):
                gt = resize_im(gt, 36)
            seg[seg == 255] = 1
            t_seg[:, :, slice_nb] = seg
            t_gt[:, :, slice_nb] = gt
        t_seg = torch.from_numpy(t_seg)
        t_gt = torch.from_numpy(t_gt)
        batch_dice[i,...] = dice_batch(class2one_hot(t_seg,3), class2one_hot(t_gt,3))[0] # do not save the interclasses etcetc
        #df = pd.DataFrame({"val_batch_dice": batch_dice})
        #df.to_csv(Path(savefolder, 'dice_3d.csv'), float_format="%.4f", index_label="epoch")
    return batch_dice.mean(dim=0), batch_dice.std(dim=0)





def metrics_calc(all_grp,all_inter_card,all_card_gt,all_card_pred, metric_axis,pprint=False):
    _, C = all_card_gt.shape
    unique_patients = torch.unique(all_grp)
    batch_dice = torch.zeros((len(unique_patients), C))
    batch_avd = torch.zeros((len(unique_patients), C))
    for i, p in enumerate(unique_patients):
        inter_card_p = torch.einsum("bc->c", [torch.masked_select(all_inter_card, all_grp == p).reshape((-1, C))])
        card_gt_p= torch.einsum("bc->c", [torch.masked_select(all_card_gt, all_grp == p).reshape((-1, C))])
        card_pred_p= torch.einsum("bc->c", [torch.masked_select(all_card_pred, all_grp == p).reshape((-1, C))])
        dice_3d = (2 * inter_card_p + 1e-8) / ((card_pred_p + card_gt_p)+ 1e-8)
        avd = (card_pred_p + card_gt_p - 2 * inter_card_p + 1e-8) / (card_gt_p + 1e-8)
        if pprint:
            dice_3d = torch.round(dice_3d * 10**2) / (10**2)
            print(p,dice_3d)
        batch_dice[i,...] = dice_3d
        batch_avd[i,...] = avd

    indices = torch.tensor(metric_axis)
    dice_3d = torch.index_select(batch_dice, 1, indices)
    avd = torch.index_select(batch_avd, 1, indices)
    dice_3d_mean = dice_3d.mean(dim=0)
    avd_mean = avd.mean(dim=0)
    print('metric_axis dice',dice_3d_mean)
    dice_3d_sd = dice_3d.std(dim=0)
    avd_sd = avd.std(dim=0)
    [dice_3d, dice_3d_sd] = map_(lambda t: t.mean(), [dice_3d_mean, dice_3d_sd])
    [avd, avd_sd] = map_(lambda t: t.mean(), [avd_mean, avd_sd])

    return dice_3d.item(), dice_3d_sd.item(), avd.item(), avd_sd.item()

if __name__ == "__main__":


    #args = Namespace(base_folder='', folders='fs_Wat_on_Inn_n', subfolders='iter000', gt_folder='data/all_transverse/train/GT/', save_folder='results/Inn/combined_adv_lambda_e1', grp_regex="Subj_\\d+_")
    #run_dices(args)

    '''
    data = 'Inn_from_server'
    subfolder = 'val'
    #
    # #DA 3d dice
    # result_fold = 'data/ivd2/'+subfolder
    # args = Namespace(base_folder='data/ivd2/'+subfolder,  folders=['fs_Wat_on_'+data], subfolders=['iter000'], gt_folder='data/ivd2/'+subfolder+'/GT/',save_folder=result_fold+'/fs_Wat_on_'+data+'_check', grp_regex="Subj_\\d+_")
    # run_dices(args)
    #
    #Constrained loss 3d dice on val
    result_fold = 'results/'+data+'/'
    methodname = next(os.walk(result_fold))[1]
    #foldernames = [result_fold + m for m in methodname]
    foldernames = [result_fold + 'presize_heaviest']

    for i, f in enumerate(foldernames):
        print(f)
        subfolders_all = next(os.walk(f))[1]
        grouping_regex = re.compile('iter')
        matches = map_(grouping_regex.search, subfolders_all)
        subfolders = [match.string+'/val/' for match in matches if type(match) != type(None)]
        args = Namespace(base_folder='', folders=[f], subfolders=subfolders, gt_folder='data/ivd2/'+subfolder+'/GT/', save_folder=f, grp_regex="Subj_\\d+_")
        run_dices(args)


    #Constrained loss 3d dice on test
    # result_fold = 'results_test/'+data+'/'
    # methodname = next(os.walk(result_fold))[1]
    #
    # for i, m in enumerate(methodname):
    #     print(m)
    #     args = Namespace(base_folder=result_fold, folders=[m], subfolders=['iter000'], gt_folder='data/ivd2/test/GT/', save_folder=result_fold+m, grp_regex="Subj_\\d+_")
    #     run_dices(args)
    
    '''




