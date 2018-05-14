import os, os.path as osp
import glob
import scipy.misc
import numpy as np

import collections

from utils import *
from Tools.transforms import *

src_dir = "Canon1DsMkIII_JPG"
mid_dir = "Canon1DsMkIII_JPG_origin_0.15"
dst_dir = "Canon1DsMkIII_JPG_disturbed"


create_if_not_exist([mid_dir, dst_dir])

ratio = 0.15
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

# iter_dirs("Canon1DsMkIII_JPG")
process = transforms.Compose([
    ScaleWithRatio(0.15),
    SaveToDir("test_saving", rename=lambda f: f.replace(".jpg", "_original.jpg")),
    PacksToTensor(),
    ApplyColorDisturb(0.6, 1.4),
    PacksToPILImage(),
    SaveToDir("test_saving", rename=lambda f: f.replace(".jpg", "_disturbed.jpg")),
])
import time
start_time = time.time()
loader = FlatImageFolder(root="Canon1DsMkIII_JPG", transform=process)
# loader = DataLoader(loader, num_workers=8, collate_fn=lambda x:x, batch_size=4)
for id, each in enumerate(loader):
    print(id, each)
    elapsed_time = time.time() - start_time
    print(elapsed_time)