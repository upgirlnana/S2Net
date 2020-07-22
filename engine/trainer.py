# encoding: utf-8


import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.reid_metric import R1_mAP

global ITER
ITER = 0

def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        #occluded training

        # img, target, occ_mask= batch
        # # # import pdb
        # # # pdb.set_trace()
        # # # print(salie_mask.size())
        # occ_mask = occ_mask.view(occ_mask.size(0), 1, occ_mask.size(1), occ_mask.size(2))
        # img = img.to(device) if torch.cuda.device_count() >= 1 else img
        # target = target.to(device) if torch.cuda.device_count() >= 1 else target
        # occ_mask = occ_mask.to(device) if torch.cuda.device_count() >= 1 else occ_mask
        # # #
        # # import pdb
        # # pdb.set_trace()
        # xishu=[]
        # for inex in range(occ_mask.size()[0]):
        #     count=0
        #     for i in range(occ_mask.size()[2]):
        #         for j in range(occ_mask.size()[3]):
        #             if occ_mask[inex,:,i,j]==1:
        #                 count=count+1
        #     a=float(count/(32*64))
        #     xishu.append(a)
        # xishu = torch.Tensor(xishu)
        # # import pdb
        # # pdb.set_trace()
        #
        #
        # score, feat ,mask= model(img)
        # loss = loss_fn(score, feat, target, mask, occ_mask,xishu)


        # body training
        # img, target, parsing_img, body_new_parsing_img = batch
        # parsing_masks = parsing_img  # recover just foreground
        # body_mask=body_new_parsing_img
        # img = img.to(device) if torch.cuda.device_count() >= 1 else img
        # target = target.to(device) if torch.cuda.device_count() >= 1 else target
        # parsing_masks=parsing_masks.to(device) if torch.cuda.device_count() >= 1 else parsing_masks
        # body_masks = body_mask.to(device) if torch.cuda.device_count() >= 1 else body_mask
        # score,feat,mask= model(img)
        # loss = loss_fn(score, feat, target,mask,parsing_masks,body_masks)
        # # #



        # salient training
        # img, target, salie_img = batch
        #   # recover just foreground
        #
        # img = img.to(device) if torch.cuda.device_count() >= 1 else img
        # target = target.to(device) if torch.cuda.device_count() >= 1 else target
        # salie_img = salie_img.to(device) if torch.cuda.device_count() >= 1 else salie_img
        # salie_img=torch.unsqueeze(salie_img,dim=1)
        # score, feat, mask = model(img)
        # loss = loss_fn(score, feat, target, mask, salie_img)
        #
        # loss.backward()
        # optimizer.step()
        # # compute acc
        # sco=score[0]+score[1]
        # acc = (score.max(1)[1] == target).float().mean()

        # both training
        imgs, target, parsing_img, body_new_parsing_img,salience_img=batch

        img = imgs.to(device) if torch.cuda.device_count() >= 1 else imgs
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        parsing_masks = parsing_img.to(device) if torch.cuda.device_count() >= 1 else parsing_img
        body_masks_true = body_new_parsing_img.to(device) if torch.cuda.device_count() >= 1 else body_new_parsing_img
        salie_img = salience_img.to(device) if torch.cuda.device_count() >= 1 else salience_img

        score, all_feat, bodymask, salietmask = model(img)
        import pdb
        pdb.set_trace()

        loss = loss_fn(score, all_feat, target, bodymask, salietmask,parsing_masks,body_masks_true,salie_img)
        loss.backward()
        optimizer.step()
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)

def fliphor(self, inputs):
    inv_idx = torch.arange(inputs.size(3) - 1, -1, -1).long()  # N x C x H x W
    return inputs.index_select(3, inv_idx)
def create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cetner_loss_weight,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        img, target,parsing_img = batch
        parsing_masks = parsing_img  # recover just foreground
        # print(parsing_masks.size())
        # import pdb
        # pdb.set_trace()
        # parsing_masks = parsing_masks.view(parsing_masks.size(0), 1, parsing_masks.size(1), parsing_masks.size(2))
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        parsing_masks = parsing_masks.to(device) if torch.cuda.device_count() >= 1 else parsing_masks
        score, feat,mask = model(img)
        # loss = loss_fn(score, feat, target)
        loss = loss_fn(score, feat, target, mask, parsing_masks)
        # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
        loss.backward()
        optimizer.step()
        for param in center_criterion.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center.step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
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
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        features = torch.FloatTensor()
        with torch.no_grad():
            #data, pids, camids ,img_path= batch
            data, pids, camids,camids ,img_path= batch
            # ff = torch.FloatTensor(data.size(0), 10240).zero_()
            # for i in range(2):
            #     if i==1:
            #         data=filter(data)
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            #     output = model(data)
            #     f = output.data.cpu()
            #     ff=ff+f
            # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            # ff = ff.div(fnorm.expand_as(ff))
            # features = torch.cat((features, ff), 0)
            feat = model(data)
            return feat, pids, camids,img_path

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)


def do_train_with_center(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer,
                                                                     'center_param': center_criterion,
                                                                     'optimizer_center': optimizer_center})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)
