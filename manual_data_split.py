import math
import os
from glob import glob
import shutil

from train import *


val_files = glob(os.path.join(TRAIN_IMG_DIR, '*'))

cutoff = math.ceil(len(val_files) * 0.8)

print(cutoff)
for train_img_path in val_files[cutoff:]:
    train_img_basename = os.path.basename(train_img_path)
    test_img_path = os.path.join(VAL_IMG_DIR, train_img_basename)
    shutil.move(train_img_path, test_img_path)

    train_m_basename = train_img_basename.replace('.jpg', '_mask.gif')
    train_m_path = os.path.join(TRAIN_MASK_DIR, train_m_basename)
    test_m_path = os.path.join(VAL_MASK_DIR, train_m_basename)
    shutil.move(train_m_path, test_m_path)
