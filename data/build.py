# encoding: utf-8


from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset, ValidImageDataset,LabelImageDataset,OccImageDataset
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms
import data.transformstwo as T

def make_data_loader(cfg):

    use_salience=True
    use_parsing=False
    print('use_salience :', use_salience)
    print('use_parsing :', use_parsing)
    transform_salience_parsing = T.Compose([
        # T.Random2DTranslation(256, 128),
        # T.RandomHorizontalFlip(),
    ])
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
        print(cfg.DATASETS.NAMES)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids

    train_set = ValidImageDataset(dataset.train, train_transforms,use_salience = use_salience,
                             use_parsing = use_parsing, salience_base_path = dataset.salience_train_dir, parsing_base_path = dataset.parsing_train_dir, transform_salience_parsing = transform_salience_parsing)
    # train_set = ImageDataset(dataset.train, cfg, use_salience=use_salience,
    #                          use_parsing=use_parsing, salience_base_path=dataset.salience_train_dir,
    #                          parsing_base_path=dataset.parsing_train_dir,
    #                          transform_salience_parsing=transform_salience_parsing)
    # if cfg.DATALOADER.SAMPLER == 'softmax':
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    val_set = ValidImageDataset(dataset.query + dataset.gallery, val_transforms)
    # val_set=LabelImageDataset(dataset.query + dataset.gallery, val_transforms,use_salience = use_salience,
    #                          use_parsing = use_parsing, salience_base_path = dataset.salience_train_dir,
    #                           parsing_base_path = dataset.parsing_train_dir, transform_salience_parsing = transform_salience_parsing)

    # query_set = LabelImageDataset(dataset.query , val_transforms, use_salience=use_salience,
    #                                                      use_parsing = use_parsing, salience_base_path = dataset.salience_query_dir,
                                                          # parsing_base_path = dataset.parsing_query_dir, transform_salience_parsing = transform_salience_parsing)

    # gallery_set = LabelImageDataset(dataset.gallery, val_transforms, use_salience=use_salience,
    #                             use_parsing=use_parsing, salience_base_path=dataset.salience_gallery_dir,
                                # parsing_base_path=dataset.parsing_gallery_dir,
                                # transform_salience_parsing=transform_salience_parsing)
    val_loader = DataLoader(val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn)
    return train_loader, val_loader, len(dataset.query), num_classes
