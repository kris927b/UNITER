"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Image Text Retrieval evaluation helper
"""
from time import time

import torch
from horovod import torch as hvd
from tqdm import tqdm

from .logger import LOGGER
from .misc import NoOp
from .distributed import all_gather_list


@torch.no_grad()
def itm_eval(score_matrix, txt_ids, img_ids, txt2img, img2txts):
    # image retrieval
    img2j = {i: j for j, i in enumerate(img_ids)}
    _, rank_txt = score_matrix.topk(10, dim=1)
    ir_r1, ir_r5, ir_r10 = 0, 0, 0
    for j, txt_id in enumerate(txt_ids):
        gt_is = [img2j[txt2img[txt_id]]]
        ranks = [(rank_txt[j, :] == i).nonzero() for i in gt_is]
        rank = min([10] + [f.item() for r in ranks for f in r if r.numel()])
        if rank < 1:
            ir_r1 += 1
        if rank < 5:
            ir_r5 += 1
        if rank < 10:
            ir_r10 += 1
    ir_r1 /= len(txt_ids)
    ir_r5 /= len(txt_ids)
    ir_r10 /= len(txt_ids)

    # text retrieval
    txt2i = {t: i for i, t in enumerate(txt_ids)}
    _, rank_img = score_matrix.topk(10, dim=0)
    tr_r1, tr_r5, tr_r10 = 0, 0, 0
    for j, img_id in enumerate(img_ids):
        gt_is = [txt2i[img2txts[img_id]]]
        ranks = [(rank_img[:, j] == i).nonzero() for i in gt_is]
        rank = min([10] + [r.item() for r in ranks if r.numel()])
        if rank < 1:
            tr_r1 += 1
        if rank < 5:
            tr_r5 += 1
        if rank < 10:
            tr_r10 += 1
    tr_r1 /= len(img_ids)
    tr_r5 /= len(img_ids)
    tr_r10 /= len(img_ids)

    tr_mean = (tr_r1 + tr_r5 + tr_r10) / 3
    ir_mean = (ir_r1 + ir_r5 + ir_r10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_log = {'txt_r1': tr_r1,
                'txt_r5': tr_r5,
                'txt_r10': tr_r10,
                'txt_r_mean': tr_mean,
                'img_r1': ir_r1,
                'img_r5': ir_r5,
                'img_r10': ir_r10,
                'img_r_mean': ir_mean,
                'r_mean': r_mean}
    return eval_log


@torch.no_grad()
def evaluate(model, eval_loader):
    st = time()
    LOGGER.info("start running Image/Text Retrieval evaluation ...")
    score_matrix = inference(model, eval_loader)
    dset = eval_loader.dataset
    all_score = hvd.allgather(score_matrix)
    all_txt_ids = [i for ids in all_gather_list(dset.ids)
                   for i in ids]
    all_img_ids = dset.all_img_ids
    assert all_score.size() == (len(all_txt_ids), len(all_img_ids))
    if hvd.rank() != 0:
        return {}

    # NOTE: only use rank0 to compute final scores
    eval_log = itm_eval(all_score, all_txt_ids, all_img_ids,
                        dset.txt2img, dset.img2txts)

    tot_time = time()-st
    LOGGER.info(f"evaluation finished in {int(tot_time)} seconds")
    return eval_log


@torch.no_grad()
def inference(model, eval_loader):
    model.eval()
    if hvd.rank() == 0:
        pbar = tqdm(total=len(eval_loader))
    else:
        pbar = NoOp()
    score_matrix = torch.zeros(len(eval_loader.dataset),
                               len(eval_loader.dataset.all_img_ids),
                               device=torch.device("cuda"),
                               dtype=torch.float16)
    for i, mini_batches in enumerate(eval_loader):
        j = 0
        for batch in mini_batches:
            scores = model(batch, compute_loss=False)
            bs = scores.size(0)
            score_matrix.data[i, j:j+bs] = scores.data.squeeze(1).half()
            j += bs
        assert j == score_matrix.size(1)
        pbar.update(1)
    model.train()
    pbar.close()
    return score_matrix
