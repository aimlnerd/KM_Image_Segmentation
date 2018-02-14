import numpy as np
import cv2
from utils.timeit_decorator import timeit

class img_preprocess():
    def __init__(self
                 ):
        pass
    def __get_im_cv2(self, path):
        # convert opencv default of BGR to RGB using [:,:,::-1] or img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.imread(path)[:, :, ::-1]
        # resized = cv2.resize(img, (img_cols, img_rows), cv2.INTER_LINEAR)
        return img

    # file = get_im_cv2('./input/train/level-3/26.tif')
    # plt.imshow(file)

    @timeit
    def __img_to_numpy(self, fpath):
        # fpath should be list of image path
        return np.array([self.__get_im_cv2(fname) for fname in fpath])

'''
loadimg = load_img(flist_train_org = flist_train_l1_org
                 ,flist_train_mask = flist_train_l1_mask
                 ,flist_train_ref = flist_train_l1_ref
                )
X_train_id, X_train_org, X_train_mask, X_train_ref = loadimg._load_train()
X_train_id.shape
X_train_org.shape
X_train_mask.shape
X_train_ref.shape
'''
