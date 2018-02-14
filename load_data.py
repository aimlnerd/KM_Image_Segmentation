import numpy as np
# np.random.seed(12022018)
import os
import cv2
from utils.timeit_decorator import timeit

class load_img():
    def __init__(self
                 , flist_train_org
                 , flist_train_mask
                 , flist_train_ref
                 , resize = None
                 , img_ext_len = 3 #length of img file extenstion tif =3, jpeg=4
                 , scale255 = False
                 ):
        self.flist_train_org = flist_train_org
        self.flist_train_mask = flist_train_mask
        self.flist_train_ref = flist_train_ref
        self.img_ext_len = img_ext_len
        self.scale255 = scale255
        self.resize = resize

    def __get_im_cv2(self, path, **kwargs):
        # convert opencv default of BGR to RGB using [:,:,::-1] or img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #key, value = kwargs.keys(), kwargs.values()

        if len(kwargs) == 1:
            if (kwargs['img_type'] == 'binary'):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #.reshape(self.resize[0],self.resize[1],1)
                #thresh = 127
                #ret, img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        elif len(kwargs) == 0:
            img = cv2.imread(path)[:, :, ::-1]
        if self.resize != None:
            img = cv2.resize(img, self.resize) #, cv2.INTER_LINEAR
        return img

    # file = get_im_cv2('./input/train/level-3/26.tif')
    # plt.imshow(file)
    #@timeit
    def __img_to_numpy(self, fpath, **kwargs):
        # fpath should be list of image path
        if len(kwargs) == 1:
            return np.array([self.__get_im_cv2(fname, img_type = kwargs['img_type']) for fname in fpath])
        elif len(kwargs) == 0:
            return np.array([self.__get_im_cv2(fname) for fname in fpath])

    #@timeit
    def load_train(self):
        X_train_org = self.__img_to_numpy(self.flist_train_org)
        Y_train_mask = self.__img_to_numpy(self.flist_train_mask,img_type = 'binary')
        X_train_ref = self.__img_to_numpy(self.flist_train_ref)

        id_list = [os.path.basename(fname)[:-(self.img_ext_len+1)] for fname in self.flist_train_org]
        X_train_id = np.array(id_list, dtype='str').reshape(len(self.flist_train_org), 1)

        if self.scale255:
            (X_train_org, Y_train_mask, X_train_ref) = (np.divide(x, 255.0) for x in [X_train_org, Y_train_mask, X_train_ref])

        return (X_train_id, X_train_org, Y_train_mask, X_train_ref)


'''
loadimg = load_img(flist_train_org = flist_train_l1_org
                 ,flist_train_mask = flist_train_l1_mask
                 ,flist_train_ref = flist_train_l1_ref
                 ,scale255 = True
                )
X_train_id, X_train_org, Y_train_mask, X_train_ref = loadimg._load_train()

X_train_id.shape
X_train_org.shape
Y_train_mask.shape
X_train_ref.shape
'''
