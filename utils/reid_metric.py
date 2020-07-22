# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking

# def get_parsedistance_matrix(X, Y,query_label_mark):
#     '''
#     Gets the distance of between each element of X and Y
#
#     input:
#
#     X: tensor of features
#     Y: tensor of features
#
#     output:
#
#     distmat: Matrix(numpy) of size |X|*|Y| with the distances between each element of X and Y
#     '''
#     sumdis=torch.zeros(300,300)
#     # sumdis = torch.zeros(1000, 1000)
#     # sumdis = torch.zeros(546, 3351)
#     # sumdis = torch.zeros(2163, 9053)
#     print('query_label_mark length',len(query_label_mark))
#     # print(query_label_mark[0])
#     # print('query_label_mark size', (query_label_mark).size())
#
#     # 300 label
#     for i in range(len(query_label_mark)):
#
#         # onedist=torch.zeros(1,1000)
#         onedist = torch.zeros(1, 300)
#         # onedist = torch.zeros(1, 3351)
#         # onedist = torch.zeros(1, 9053)
#         # each label 6 part
#         for j in range(6):
#             # index=label[j]
#             m,n=1,300
#             # m, n = 1, 9053
#             # m, n = 1, 1000
#             # m, n = 1, 3351
#             # import pdb
#             # pdb.set_trace()
#             # a=torch.pow(torch.unsqueeze(X[i][j*2048:((j+1)*2048)-1],0), 2).sum(dim=1, keepdim=True)
#             # b=torch.pow(Y[:,j*2048:((j+1)*2048)-1], 2).sum(dim=1, keepdim=True).expand(n, m).t()
#             tempdist=0
#             count=0
#             if query_label_mark[i][j] > 0:
#                 count=count+1
#                 tempdist=torch.pow(torch.unsqueeze(X[i][j*2048:((j+1)*2048)-1],0), 2).sum(dim=1, keepdim=True).expand(m, n) + \
#               torch.pow(Y[:,j*2048:((j+1)*2048)-1], 2).sum(dim=1, keepdim=True).expand(n, m).t()
#
#                 tempdist.addmm_(1, -2, torch.unsqueeze(X[i][j*2048:((j+1)*2048)-1],0), Y[:,j*2048:((j+1)*2048)-1].t())
#                 # tempdist=tempdist.numpy()
#
#             onedist=onedist+tempdist
#         sumdis[i,:]=(onedist)
#     # sumdis=torch.tensor(sumdis)
#     # return  sumdis
#
#
#     # m, n = X.size(0), Y.size(0)
#     # distmat = torch.pow(X, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#     #           torch.pow(Y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#     # distmat.addmm_(1, -2, X, Y.t())
#     # # distmat[distmat<0]=0
#     # distmat=np.sqrt(distmat)
#     # sumdis = sumdis.numpy()
#     print(len(sumdis))
#     return sumdis.numpy()
def get_parsedistance_matrix(X, Y,query_label_mark):
    '''
    Gets the distance of between each element of X and Y

    input:

    X: tensor of features
    Y: tensor of features

    output:

    distmat: Matrix(numpy) of size |X|*|Y| with the distances between each element of X and Y
    '''
    #
    # sumdis=torch.zeros(300,300)
    sumdis = torch.zeros(1000, 1000)
    sumdis = torch.zeros(119, 119)
    print(len(query_label_mark))
    for i in range(0,len(query_label_mark)):
        label=np.unique(query_label_mark[i])
        # onedist=torch.zeros(1,300)
        onedist = torch.zeros(1, 1000)
        onedist = torch.zeros(1, 119)

        for j in range(0,len(label)):
            index=label[j]
            # m,n=1,300
            # m, n = 1, 1000
            m, n = 1, 119
            if index==0:
                print()
                # tempdist=torch.zeros(1,1000)
                tempdist = torch.zeros(1, 119)
              #   tempdist=torch.pow(torch.unsqueeze(X[i][0:2047],0), 2).sum(dim=1, keepdim=True).expand(m, n) + \
              # torch.pow(Y[:,0:2047], 2).sum(dim=1, keepdim=True).expand(n, m).t()
              #   tempdist.addmm_(1, -2, torch.unsqueeze(X[i][0:2047],0), Y[:,0:2047].t())
            elif index==1:
                # print((X[i][2049:3072].size()))
                # print(type(X[i][2049:3072]))
                # print((Y[:, 2049:3072]).size())
                # print(type(Y[:, 2049:3072]))
                # import pdb
                # pdb.set_trace()
                tempdist = torch.pow(torch.unsqueeze(X[i][2048:4095],0), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                           torch.pow(Y[:, 2048:4095], 2).sum(dim=1, keepdim=True).expand(n, m).t()
                tempdist.addmm_(1, -2, torch.unsqueeze(X[i][2048:4095],0), Y[:, 2048:4095].t())
            elif index==2:
                tempdist = torch.pow(torch.unsqueeze(X[i][4096:6143],0), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                           torch.pow(Y[:, 4096:6143], 2).sum(dim=1, keepdim=True).expand(n, m).t()
                tempdist.addmm_(1, -2, torch.unsqueeze(X[i][4096:6143],0), Y[:, 4096:6143].t())
            elif index==3:
                tempdist = torch.pow(torch.unsqueeze(X[i][6144:8191],0), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                           torch.pow(Y[:, 6144:8191], 2).sum(dim=1, keepdim=True).expand(n, m).t()
                tempdist.addmm_(1, -2, torch.unsqueeze(X[i][6144:8191],0), Y[:, 6144:8191].t())
            else:
                tempdist = torch.pow(torch.unsqueeze(X[i][8192:10239], 0), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                           torch.pow(Y[:, 8192:10239], 2).sum(dim=1, keepdim=True).expand(n, m).t()
                tempdist.addmm_(1, -2, torch.unsqueeze(X[i][8192:10239], 0), Y[:, 8192:10239].t())
            # import pdb
            # pdb.set_trace()
            onedist=onedist.cpu()+tempdist.cpu()

        sumdis[i,:]=onedist
    # sumdis=torch.tensor(sumdis)
    # return  sumdis


    # m, n = X.size(0), Y.size(0)
    # distmat = torch.pow(X, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #           torch.pow(Y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # distmat.addmm_(1, -2, X, Y.t())
    # # distmat[distmat<0]=0
    # distmat=np.sqrt(distmat)
    # sumdis = sumdis.numpy()
    print(len(sumdis))
    return sumdis.numpy()

class Rank_image(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(Rank_image, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.feats1 = []
        self.feats2 = []
        self.pids = []
        self.camids = []
        self.labels = []
        self.img_path=[]

    def update(self, output):
        # feat, pid, camid = output
        # feat, feat1, feat2,bodymap,salientmap, pid, camid, label ,img_path= output
        feat, feat1, feat2, pid, camid, label, img_path = output
        # import pdb
        # pdb.set_trace()
        self.feats.append(feat)
        self.feats1.append(feat1)
        self.feats2.append(feat2)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.labels.extend(label)
        self.img_path.extend(img_path)

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        feats1 = torch.cat(self.feats1, dim=0)
        feats2 = torch.cat(self.feats2, dim=0)
        label = self.labels
        img_path=self.img_path
        # label=torch.cat(self.labels,dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            feats1 = torch.nn.functional.normalize(feats1, dim=1, p=2)
            feats2 = torch.nn.functional.normalize(feats2, dim=1, p=2)
        # query
        qf = feats[:self.num_query]

        qf1 = feats1[:self.num_query]
        qf2 = feats2[:self.num_query]
        label = label[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        query_img_paths=np.asarray(img_path[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        gf1 = feats1[self.num_query:]
        gf2 = feats2[self.num_query:]

        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        gallery_img_paths =np.asarray(img_path[self.num_query:])
        # import pdb
        # pdb.set_trace()
        m, n = qf1.shape[0], gf1.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        distmat1 = torch.pow(qf1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf1, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat2 = torch.pow(qf2, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat1.addmm_(1, -2, qf1, gf1.t())
        distmat2.addmm_(1, -2, qf2, gf2.t())
        distmat = distmat.cpu().numpy()
        distmat1 = distmat1.cpu().numpy()
        distmat2 = distmat2.cpu().numpy()
        distmat = (1 - 0.4) * distmat1 + 0.4* distmat2
        ranks = [1, 3, 5, 10]
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids,query_img_paths,gallery_img_paths, max_rank=50,save_dir='/home/rxn/myproject/MGAN/pth/fig/OCCLUDED/occduke_nocolor',epoch=-1,save_rank=True)
        # for lam in range(0, 11):
        #     print(lam)
        #     weight = lam * 0.1
        # #     # print(type(weight))
        # #     # print(type(distmat))
        # #     # print(type(parse_distmat))
        # #     # print(type(globaldist))
        #     distmat = (1 - weight) * distmat1 + weight * distmat2
        # #     # distmat = (1 - weight) * torch.from_numpy(parse_distmat)+  weight*torch.from_numpy(distmat2) \
        # #     #           + weight * globaldist.cpu()
        # #     #           # +globaldist.cpu()
        # #     # + weight * globaldist.cpu()+\
        # #     # weight*torch.from_numpy(distmat2)
        # #
        #     cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, query_img_paths, gallery_img_paths,
        #                          max_rank=50, save_dir='/home/rxn/myproject/MGAN/pth/fig/duke', epoch=-1,
        #                          save_rank=False)
        # #     # cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        # # #     # wcmc, wmAP = eval_func(globaldist.cpu(), q_pids, g_pids, q_camids, g_camids)
        # # #     # hcmc, hmAP = eval_func(headdist.cpu(), q_pids, g_pids, q_camids, g_camids)
        # # #     # ucmc, umAP = eval_func(upperdist.cpu(), q_pids, g_pids, q_camids, g_camids)
        # # #     # lcmc, lmAP = eval_func(lowerdist.cpu(), q_pids, g_pids, q_camids, g_camids)
        # #     # scmc, smAP = eval_func(shoesdist.cpu(), q_pids, g_pids, q_camids, g_camids)
        #     print("Results ----------")
        #     print("mAP: {:.2%}".format(mAP))
        # #     # print("hmAP: {:.2%}".format(hmAP))
        # #     # print("umAP: {:.2%}".format(umAP))
        # #     # print("lmAP: {:.2%}".format(lmAP))
        # #     # print("smAP: {:.2%}".format(smAP))
        # # #
        #     print("CMC curve")
        #     for r in ranks:
        #         print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))
        #         # print("hRank-{:<3}: {:.2%}".format(r, hcmc[r - 1]))
        #         # print("uRank-{:<3}: {:.2%}".format(r, ucmc[r - 1]))
        #         # print("lRank-{:<3}: {:.2%}".format(r, lcmc[r - 1]))
        #         # print("sRank-{:<3}: {:.2%}".format(r, scmc[r - 1]))
        #
        #     print("------------------")

        return cmc, mAP


class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.img_path = []

    def update(self, output):
        feat, pid, camid,img_path = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.img_path.extend(img_path)


    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        img_path = self.img_path
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        query_img_paths = np.asarray(img_path[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        gallery_img_paths = np.asarray(img_path[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        # globaldist = torch.pow(qf1[:, 0:2047], 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #              torch.pow(gf1[:, 0:2047], 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # globaldist.addmm_(1, -2, qf1[:, 0:2047], gf1[:, 0:2047].t())
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids,query_img_paths,gallery_img_paths, max_rank=50,save_dir = "/home/rxn/myproject/MGAN/pth/fig/OCCLUDED/P_DUKE",epoch = -1,save_rank =False)

        return cmc, mAP



class Two_R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(Two_R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.feats1 = []
        self.feats2 = []
        self.pids = []
        self.camids = []
        self.labels=[]

    def update(self, output):
        # feat, pid, camid = output
        feat,feat1, feat2,pid, camid,label = output
        # import pdb
        # pdb.set_trace()
        self.feats.append(feat)
        self.feats1.append(feat1)
        self.feats2.append(feat2)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.labels.extend(label)

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        feats1 = torch.cat(self.feats1, dim=0)
        feats2 = torch.cat(self.feats2, dim=0)
        label=self.labels
        # label=torch.cat(self.labels,dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            feats1 = torch.nn.functional.normalize(feats1, dim=1, p=2)
            feats2 = torch.nn.functional.normalize(feats2, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        qf1 = feats1[:self.num_query]
        qf2 = feats2[:self.num_query]
        label=label[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        gf1 = feats1[self.num_query:]
        gf2 = feats2[self.num_query:]

        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # import pdb
        # pdb.set_trace()
        m, n = qf1.shape[0], gf1.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # globaldist = torch.pow(qf1[:,0:2047], 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #     torch.pow(gf1[:,0:2047], 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # globaldist.addmm_(1, -2, qf1[:,0:2047], gf1[:,0:2047].t())
        #
        # headdist = torch.pow(qf1[:,2048:4095], 0).sum(dim=1, keepdim=True).expand(m, n) + \
        #            torch.pow(gf1[:, 2048:4095], 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # headdist.addmm_(1, -2, qf1[:,2048:4095], gf1[:, 2048:4095].t())
        #
        # upperdist = torch.pow(qf1[:,4096:6143], 0).sum(dim=1, keepdim=True).expand(m, n) + \
        #            torch.pow(gf1[::, 4096:6143], 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # upperdist.addmm_(1, -2, qf1[:,4096:6143], gf1[:,  4096:6143].t())
        #
        # lowerdist = torch.pow(qf1[:,6144:8191], 0).sum(dim=1, keepdim=True).expand(m, n) + \
        #            torch.pow(gf1[:, 6144:8191], 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # lowerdist.addmm_(1, -2, qf1[:,6144:8191], gf1[:, 6144:8191].t())


        # shoesdist = torch.pow(qf1[:,8192:10239], 0).sum(dim=1, keepdim=True).expand(m, n) + \
        #            torch.pow(gf1[:, 8192:10239], 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # shoesdist.addmm_(1, -2, qf1[:,8192:10239], gf1[:, 8192:10239].t())

        # parse_distmat=get_parsedistance_matrix(qf1,gf1,label)
        # import pdb
        # pdb.set_trace()

        distmat1 = torch.pow(qf1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf1, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat2 = torch.pow(qf2, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat1.addmm_(1, -2, qf1, gf1.t())
        distmat2.addmm_(1, -2, qf2, gf2.t())
        distmat = distmat.cpu().numpy()
        distmat1 = distmat1.cpu().numpy()
        distmat2 = distmat2.cpu().numpy()
        ranks = [1, 3, 5, 10]
        cmc, mAP = eval_func(distmat1, q_pids, g_pids, q_camids, g_camids)
        for lam in range(0, 11):
            print(lam)
            weight = lam * 0.1
            # print(type(weight))
            # print(type(distmat))
            # print(type(parse_distmat))
            # print(type(globaldist))
            distmat=(1-weight)*distmat1+weight*distmat2
            # distmat = (1 - weight) * torch.from_numpy(parse_distmat)+  weight*torch.from_numpy(distmat2) \
            #           + weight * globaldist.cpu()
            #           # +globaldist.cpu()
                      # + weight * globaldist.cpu()+\
            # weight*torch.from_numpy(distmat2)

            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
            # wcmc, wmAP = eval_func(globaldist.cpu(), q_pids, g_pids, q_camids, g_camids)
            # hcmc, hmAP = eval_func(headdist.cpu(), q_pids, g_pids, q_camids, g_camids)
            # ucmc, umAP = eval_func(upperdist.cpu(), q_pids, g_pids, q_camids, g_camids)
            # lcmc, lmAP = eval_func(lowerdist.cpu(), q_pids, g_pids, q_camids, g_camids)
            # scmc, smAP = eval_func(shoesdist.cpu(), q_pids, g_pids, q_camids, g_camids)
            print("Results ----------")
            print("mAP: {:.2%}".format(mAP))
            # print("hmAP: {:.2%}".format(hmAP))
            # print("umAP: {:.2%}".format(umAP))
            # print("lmAP: {:.2%}".format(lmAP))
            # print("smAP: {:.2%}".format(smAP))

            print("CMC curve")
            for r in ranks:
                print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))
                # print("hRank-{:<3}: {:.2%}".format(r, hcmc[r - 1]))
                # print("uRank-{:<3}: {:.2%}".format(r, ucmc[r - 1]))
                # print("lRank-{:<3}: {:.2%}".format(r, lcmc[r - 1]))
                # print("sRank-{:<3}: {:.2%}".format(r, scmc[r - 1]))


            print("------------------")

        return cmc, mAP



class Two_R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(Two_R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):

        self.feats = []
        self.feats1 = []
        self.feats2 = []
        self.pids = []
        self.camids = []
        self.labels = []

    def update(self, output):
        # feat, pid, camid = output
        feat, feat1, feat2, pid, camid,label = output
        self.feats.append(feat)
        self.feats1.append(feat1)
        self.feats2.append(feat2)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        feats1 = torch.cat(self.feats1, dim=0)
        feats2 = torch.cat(self.feats2, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            feats1 = torch.nn.functional.normalize(feats1, dim=1, p=2)
            feats2 = torch.nn.functional.normalize(feats2, dim=1, p=2)

        # query
            # query
            qf = feats[:self.num_query]
            qf1 = feats1[:self.num_query]
            qf2 = feats2[:self.num_query]
            q_pids = np.asarray(self.pids[:self.num_query])
            q_camids = np.asarray(self.camids[:self.num_query])
            # gallery
            gf = feats[self.num_query:]
            gf1 = feats1[self.num_query:]
            gf2 = feats2[self.num_query:]

            g_pids = np.asarray(self.pids[self.num_query:])
            g_camids = np.asarray(self.camids[self.num_query:])
            m, n = qf1.shape[0], gf1.shape[0]
            edu_distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat1 = torch.pow(qf1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                       torch.pow(gf1, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat2 = torch.pow(qf2, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                       torch.pow(gf2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            edu_distmat.addmm_(1, -2, qf, gf.t())
            distmat1.addmm_(1, -2, qf1, gf1.t())
            distmat2.addmm_(1, -2, qf2, gf2.t())
            edu_distmat = edu_distmat.cpu().numpy()
            distmat1 = distmat1.cpu().numpy()
            distmat2 = distmat2.cpu().numpy()
            ranks = [1, 3, 5, 10]
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        print("Results ----------")
        print("mAP: {:.2%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))
        print("------------------")

        return cmc, mAP
class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP