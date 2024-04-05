import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torch
import torch.nn as nn
from collections import defaultdict

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import pickle
import torchvision
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.cuda.amp import autocast, GradScaler
import argparse
from PIL import Image
import xml.etree.ElementTree as ET
from transform import box2hm
from utils import get_normalizer
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from utils import save_pickle, load_pickle
import torch
import random
from hourglass import StackedHourglass
from loss import LossCalculator
from optim import get_optimizer
from data import load_dataset
from utils import AverageMeter, blend_heatmap

from torchsummary import summary
from utils import load_pickle, save_pickle
from transform import hm2box
from utils import get_normalizer
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from data import load_dataset
from train import load_network
from utils import AverageMeter, save_pickle
from transform import hm2box
from datetime import timedelta

from config import get_arguments
from train import distributed_device_train
from evaluate import single_device_evaluate

if __name__ == '__main__':
    args = get_arguments()

    tictoc = time.time()
    if args.train_flag:
        distributed_device_train(args)
    else:
        single_device_evaluate(args)
    print('%s: Process is Done During %s'%(time.ctime(), str(timedelta(seconds=(time.time() - tictoc)))))
