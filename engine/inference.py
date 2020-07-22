# encoding: utf-8

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine

from utils.reid_metric import R1_mAP, R1_mAP_reranking,Two_R1_mAP,Two_R1_mAP_reranking,Rank_image

def create_supervised_evaluator(model1,model2, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model1 = nn.DataParallel(model1)
            model2 = nn.DataParallel(model2)
        model1.to(device)
        model2.to(device)

    def _inference(engine, batch):
        model1.eval()
        model2.eval()
        with torch.no_grad():
            data, pids, camids,label,img_path = batch
            # data, pids, camids, label = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            # feat1,bodymap = model1(data)
            # feat2,salientmap=model2(data)
            feat1= model1(data)
            feat2= model2(data)
            feat=torch.cat((feat1,feat2),dim=1)
            print(feat.size())
            # return feat,feat1,feat2,bodymap,salientmap, pids, camids,label,img_path
            return feat,feat1,feat2,pids, camids,label,img_path

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model1,
        model2,
        val_loader,
        num_query
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(model1,model2, metrics={'r1_mAP': Rank_image(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model1,model2, metrics={'r1_mAP': Two_R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))

    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1,2,3,4,5,6,7,8,9,10]:
        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
