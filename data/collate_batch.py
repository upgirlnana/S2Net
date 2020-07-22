# encoding: utf-8


import torch


def train_collate_fn(batch):
    imgs, pids, _, parsing_img,body_new_parsing_img,_, = zip(*batch)
    # imgs, pids, _, occ_img,  _, = zip(*batch)
   # imgs, pids, _, salie_img,  _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    # return torch.stack(imgs, dim=0), pids,torch.stack(occ_img, dim=0)
    #return torch.stack(imgs, dim=0), pids, torch.stack(salie_img, dim=0)
    return torch.stack(imgs, dim=0), pids,torch.stack(parsing_img, dim=0),torch.stack(body_new_parsing_img, dim=0)


def val_collate_fn(batch):
    imgs, pids, camids,camids, img_path= zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids,camids,img_path
    # imgs, pids, camids,label, img_path = zip(*batch)
    # return torch.stack(imgs, dim=0), pids, camids,label,img_path
    #
