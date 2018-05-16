import os, os.path as osp
import glob
import scipy.misc
import cv2
import numpy as np

import collections

src_dir = "Canon1DsMkIII_JPG"
mid_dir = src_dir + "_origin"
dst_dir = src_dir + "_disturbed"
rec_dir = src_dir + "_recovered"
noi_dir = rec_dir = src_dir + "_noise"


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


create_if_not_exist([mid_dir, dst_dir, rec_dir, noi_dir])

resize_ratio = 0.15

t_global = 0.5


def run_one_images(f):
    if not (f.endswith(".jpg") or f.endswith(".jpeg")):
        return
    ipath = osp.join(root, f)
    img = scipy.misc.imread(ipath)
    print(ipath, img.shape)
    img = scipy.misc.imresize(img, resize_ratio)
    w, h, c = img.shape
    nw = w // 8 * 8
    nh = h // 8 * 8
    img = scipy.misc.imresize(img, (nw, nh))
    img = img.astype(np.float)
    print(img.shape)
    noise = np.ones_like(img) * 127

    try:
        img[0, 0, 2]
    except IndexError:
        return
    scipy.misc.imsave(osp.join(mid_dir, f.replace(".", "-%d." % notion)), img)
    return
    if np.random.uniform(-0.5, 0.5) > 0:
        # if 1 > 0:
        ratio = 0.5
        disturb = np.random.uniform(0.6, 1.4)
        # disturb = ratio
        img[:, :, 0] *= disturb
        noise[:, :, 0] *= disturb

        disturb = np.random.uniform(0.6, 1.4)
        # disturb = ratio
        # img[:, :, 1] *= disturb
        img[:, :, 2] *= disturb
        noise[:, :, 2] *= disturb
    else:
        a1 = np.random.uniform(0.4, 0.6)
        a2 = np.random.uniform(1.4, 1.6)
        tmin = min(a1, a2)
        tmax = max(a1, a2)
        w, h, c = img.shape

        steps = (tmax - tmin) / h
        temp = np.arange(tmin, tmax, steps)
        if len(temp) == h - 1:
            temp = np.concatenate([temp, [a2]])
        elif len(temp) == h + 1:
            temp = temp[:-1]
        if np.random.uniform(-0.5, 0.5) > 0:
            temp = temp[::-1]
        disturb = np.diag(temp)
        img[:, :, 0] = np.dot(img[:, :, 0], disturb)
        noise[:, :, 0] = np.dot(noise[:, :, 0], disturb)

        a1 = np.random.uniform(0.4, 0.6)
        a2 = np.random.uniform(1.4, 1.6)
        tmin = min(a1, a2)
        tmax = max(a1, a2)
        w, h, c = img.shape
        steps = (tmax - tmin) / h
        temp = np.arange(tmin, tmax, steps)
        if len(temp) == h - 1:
            temp = np.concatenate([temp, [a2]])
        elif len(temp) == h + 1:
            temp = temp[:-1]
        if np.random.uniform(-0.5, 0.5) > 0:
            temp = temp[::-1]
        disturb = np.diag(temp)
        img[:, :, 2] = np.dot(img[:, :, 2], disturb)
        noise[:, :, 2] = np.dot(noise[:, :, 2], disturb)

    # img[img > 255] = 255
    # img[img < 0] = 0
    # noise[noise > 255] = 255
    # noise[noise < 0] = 0
    if (img > 255).sum() > 0:
        maxV = img.max()
        img = img / maxV

    # noise = np.ones_like(img)  * 127
    scipy.misc.imsave(osp.join(dst_dir, f.replace(".", "-%d." % notion)), img)
    scipy.misc.imsave(osp.join(noi_dir, f.replace(".", "-%d." % notion)), noise)


def opencv_save(folder, image):
    from scipy.misc import imsave
    # srcRGB = np.copy(image)
    # destRGB = cv2.cvtColor(srcRGB, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(folder, destRGB)
    imsave(folder, image.astype(np.uint8))


for notion in range(1, 8):
    for root, dirs, filenames in os.walk(src_dir):
        for f in filenames:
            if not (f.endswith(".jpg") or f.endswith(".jpeg")):
                continue
            ipath = osp.join(root, f)
            img = scipy.misc.imread(ipath)
            print(ipath, img.shape)
            img = scipy.misc.imresize(img, resize_ratio)
            w, h, c = img.shape
            nw = w // 8 * 8
            nh = h // 8 * 8
            img = scipy.misc.imresize(img, (nw, nh))
            img = img.astype(np.float)
            print(img.shape)
            noise = np.ones_like(img) * 127

            try:
                img[0, 0, 2]
            except IndexError:
                continue

            opencv_save(osp.join(mid_dir, f.replace(".", "-%d." % notion)).replace(".jpg", ".png"), img)

            if np.random.uniform(-0.5, 0.5) > 0:
                # if 1 > 0:
                ratio = 0.5
                disturb = np.random.uniform(0.6, 1.4)
                # disturb = ratio
                img[:, :, 0] *= disturb
                noise[:, :, 0] *= disturb

                disturb = np.random.uniform(0.6, 1.4)
                # disturb = ratio
                # img[:, :, 1] *= disturb
                img[:, :, 2] *= disturb
                noise[:, :, 2] *= disturb
            else:
                a1 = np.random.uniform(0.4, 0.6)
                a2 = np.random.uniform(1.4, 1.6)
                tmin = min(a1, a2)
                tmax = max(a1, a2)
                w, h, c = img.shape

                steps = (tmax - tmin) / h
                temp = np.arange(tmin, tmax, steps)
                if len(temp) == h - 1:
                    temp = np.concatenate([temp, [a2]])
                elif len(temp) == h + 1:
                    temp = temp[:-1]
                if np.random.uniform(-0.5, 0.5) > 0:
                    temp = temp[::-1]
                disturb = np.diag(temp)
                img[:, :, 0] = np.dot(img[:, :, 0], disturb)
                noise[:, :, 0] = np.dot(noise[:, :, 0], disturb)

                a1 = np.random.uniform(0.4, 0.6)
                a2 = np.random.uniform(1.4, 1.6)
                tmin = min(a1, a2)
                tmax = max(a1, a2)
                w, h, c = img.shape
                steps = (tmax - tmin) / h
                temp = np.arange(tmin, tmax, steps)
                if len(temp) == h - 1:
                    temp = np.concatenate([temp, [a2]])
                elif len(temp) == h + 1:
                    temp = temp[:-1]
                if np.random.uniform(-0.5, 0.5) > 0:
                    temp = temp[::-1]
                disturb = np.diag(temp)
                img[:, :, 2] = np.dot(img[:, :, 2], disturb)
                noise[:, :, 2] = np.dot(noise[:, :, 2], disturb)

            # img[img > 255] = 255
            # img[img < 0] = 0
            # noise[noise > 255] = 255
            # noise[noise < 0] = 0
            if (img > 255).sum() > 0:
                maxV = img.max()
                img = img / (maxV / 255.0)
            # noise = np.ones_like(img)  * 127
            opencv_save(osp.join(dst_dir, f.replace(".", "-%d." % notion)).replace(".jpg", ".png"), img)
            opencv_save(osp.join(noi_dir, f.replace(".", "-%d." % notion)).replace(".jpg", ".png"), noise)

            Rmin = np.min(noise[:, :, 0])
            Gmin = np.min(noise[:, :, 1])
            Bmin = np.min(noise[:, :, 2])
            print("Rmin:%.2f Gmin:%.2f Bmin:%.2f\n" %
                  (Rmin, Gmin, Bmin)
                  )
            assert Rmin > 0
            assert Gmin > 0
            assert Bmin > 0
