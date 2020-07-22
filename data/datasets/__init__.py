# encoding: utf-8

from .occludeduke import OccDukeMTMCreID
from .cuhk03_np import  CUHK03_NP
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .veri import VeRi
from .ethz import P_ETHZ
from .p_duek import P_DUKE
from .partialreid import PartialREID
from .occluded import Occluded
from .iLIDS import PartialLIDS
from .dataset_loader import ImageDataset,ValidImageDataset,LabelImageDataset,OccImageDataset

__factory = {
    'p_duke':P_DUKE,
    'occduke':OccDukeMTMCreID,
  }


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
