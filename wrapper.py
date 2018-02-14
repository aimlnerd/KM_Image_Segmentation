
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

import load_data
reload(load_data)
from load_data import load_img

INPUT_PATH = './input'
TRAIN_DATA = os.path.join(INPUT_PATH, "train")
DEV_DATA = os.path.join(INPUT_PATH, "dev")
level = "level-1"

flist_train_l1_org = glob.glob(TRAIN_DATA + "/" + level + "/*[0-9].tif")
flist_train_l1_mask = glob.glob(TRAIN_DATA + "/" + level + "/*[0-9]_mask.tif")
flist_train_l1_ref = glob.glob(TRAIN_DATA + "/" + level + "/*[0-9]_ref.tif")

loadimg = load_img(flist_train_org = flist_train_l1_org
,flist_train_mask = flist_train_l1_mask
,flist_train_ref = flist_train_l1_ref
,scale255 = True
,resize = (128, 128)
)
X_train_id, X_train_org, Y_train_mask, X_train_ref = loadimg.load_train()

X_train_org.shape
'''
del(X_train_ref)
del(X_train_org)
del(Y_train_mask)

'''
id = 2
print(X_train_id[id])
plt.imshow(X_train_org[id,:])
plt.imshow(Y_train_mask[id,:])
plt.imshow(X_train_ref[id,:])
