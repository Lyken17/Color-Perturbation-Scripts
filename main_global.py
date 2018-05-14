import os, os.path as osp
import glob
import scipy.misc
import numpy as np

import collections

dst = "Canon1DsMkIII_JPG"
src_dir = "%s" % dst
mid_dir = "%s_origin" % dst
dst_dir = "%s_disturbed" % dst
tmp_dir = "%s_shift" % dst

def create_if_not_exist(temp):
    if isinstance(temp, str):
        if not os.path.exists(temp):
            os.makedirs(temp)
    elif isinstance(temp, collections.Iterable):
        for each in temp:
            if not os.path.exists(each):
                os.makedirs(each)
    else:
        raise NotImplementedError

create_if_not_exist([mid_dir, dst_dir, tmp_dir])


ratio = 0.15
for notion in range(1, 4):
    for root, dirs, filenames in os.walk(src_dir):
        for f in filenames:
            if not (f.endswith(".jpg") or f.endswith(".jpeg")):
                continue
            ipath = osp.join(root, f)
            print(ipath)
            img = scipy.misc.imread(ipath)
            img = scipy.misc.imresize(img, ratio)
            img = img.astype(np.float)
            print(img.shape)

            tmp = np.ones_like(img) * 128
            try:
                img[0, 0, 2]
            except IndexError:
                continue
            scipy.misc.imsave(osp.join(mid_dir,  f.replace(".", "-%d." % notion)), img)

            disturb = np.random.uniform(0.6, 1.4)
            tmp[:, :, 0] *= disturb
            img[:, :, 0] *= disturb

            disturb = np.random.uniform(0.6, 1.4)
            tmp[:, :, 2] *= disturb
            img[:, :, 2] *= disturb

            img[img > 255] = 255
            img[img < 0] = 0

            tmp[tmp > 255] = 255
            tmp[tmp < 0] = 0

            scipy.misc.imsave(osp.join(dst_dir,  f.replace(".", "-%d." % notion)), img)
            scipy.misc.imsave(osp.join(tmp_dir,  f.replace(".", "-%d." % notion)), tmp)
