import torch.utils.data as data
import os
import os.path
import numpy as np
import torch
import dicom

#root_dir = r'/AD_NC_data'  #the path in the docker container that the volume('\\10.81.20.142\finalADNI') hang up to

#184/270.270
nDepth = 184
nWidth = 268
nHeight = 268

nc_train = 969
ad_train = 535
nc_test = 245
ad_test = 135

class ADNI(data.Dataset):

    def __init__(self, is_train=True, is_label_input=False, is_autoencoder=True, root_dir=r'/AD_NC_data', transform=None, target_transform=None):
        self.transform = transform
        self.target_transform=target_transform
        self.is_train = is_train  # training set or test set
        self.is_label_input = is_label_input # just label or voxel value of parcel
        self.is_AE = is_autoencoder
        self.root_dir = root_dir
        self.NC_list = sorted(os.listdir(os.path.join(self.root_dir, 'NC')))
        self.AD_list = sorted(os.listdir(os.path.join(self.root_dir, 'AD')))
        if self.is_train:
            self.data, self.labels = self.get_train()
        else:
            self.data, self.labels = self.get_test()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        #filename = self.data[index]
        target = self.labels[index]
        image = self.get_each_raw_data(index)
        image = self.TmpNormalization(image)
        image = torch.from_numpy(image)


        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.data)

    def TmpNormalization(self, x):
        tmp = x
        if self.is_label_input:
            tmp[tmp > 0] = 1
            tmp = np.array(tmp, dtype=np.float32)
        else:
            tmp = np.array(x/2000.0, dtype=np.float32)
        return tmp

    def MaxMinNormalization(self, x, filename):
        if np.max(x) == np.min(x):
            bb = np.array(np.zeros((nDepth, nHeight, nWidth)), dtype=np.float32)
            return bb
        else:
            try:
                bb = np.array((x - np.min(x)) / (np.max(x) - np.min(x)), dtype=np.float32)
            except:
                print('except...')
                print(filename)
            else:
                return bb

    def get_train(self):
        train_NC_parcel_list = []
        train_AD_parcel_list = []
        train_label=[]
        for i in range(nc_train):
            train_NC_parcel_list.append(self.NC_list[i])
            train_label.append(0)
        for i in range(ad_train):
            train_AD_parcel_list.append(self.AD_list[i])
            train_label.append(1)
        train_list = train_NC_parcel_list + train_AD_parcel_list
        return train_list, train_label


    def get_test(self):
        test_NC_parcel_list = []
        test_AD_parcel_list = []
        test_label=[]
        for i in range(nc_train, nc_train+nc_test):
            test_NC_parcel_list.append(self.NC_list[i])
            test_label.append(0)
        for i in range(ad_train, ad_train+ad_test):
            test_AD_parcel_list.append(self.AD_list[i])
            test_label.append(1)
        test_list = test_NC_parcel_list + test_AD_parcel_list
        return test_list, test_label

    def get_each_raw_data(self, index):
        filename = self.data[index]
        NC_raw_dir = os.path.join(self.root_dir, 'NC')
        AD_raw_dir = os.path.join(self.root_dir, 'AD')
        if self.is_label_input:
            NC_raw_dir = os.path.join(self.root_dir, 'NC_sorted_label')
            AD_raw_dir = os.path.join(self.root_dir, 'AD_sorted_label')
        if self.labels[index]==0:
            filepath = os.path.join(NC_raw_dir, filename)
            #prefix = ''
        else:
            filepath = os.path.join(AD_raw_dir, filename)
            #prefix='labels_'
        name_list = sorted(os.listdir(filepath))
        each_data = []

        '''
        label_data = []
        for m in range(len(name_list)):
            label_name = prefix + str(m).zfill(5) + '.dcm'
            try:
                label_data = dicom.read_file(os.path.join(filepath, label_name))
            except:
                print(os.path.join(filepath, label_name))
            label_data = label_data.pixel_array
            label_data_2 = np.zeros(label_data.shape, dtype=np.float32)
            label_data_2[:, :] = label_data[:, :]
            each_data.append(label_data_2)

        '''
        dcm_dir = sorted(name_list, key=lambda x: int(x.split('_')[-3]))   #数据可能按照1 10 100这样的顺序，不对，所以需要重新排序
        for filename in dcm_dir:
            if '.dcm' not in filename:
                continue
            ds = dicom.read_file(os.path.join(filepath, filename)).pixel_array
            data = np.zeros(ds.shape, dtype=np.float32)
            data[:, :] = ds[:, :]
            each_data.append(data)
        return self.add_boundingbox(np.array(each_data))

    def add_boundingbox(self, arr):

        arr_2 = np.zeros((nDepth, nHeight, nWidth), dtype=np.float32)
        d, h, w = arr.shape
        d_from = h_from = w_from = 0

        d1 = nDepth//2 - d // 2
        if d1 < 0:
            d_from = d // 2 - nDepth//2
            d = nDepth
            d1 = 0
        h1 = nHeight//2 - h // 2
        if h1 < 0:
            h_from = h // 2 - nHeight//2
            h = nHeight
            h1 = 0
        w1 = nWidth//2 - w // 2
        if w1 < 0:
            w_from = w // 2 - nWidth//2
            w = nWidth
            w1 = 0


        arr_2[d1:d1 + d, h1:h1 + h, w1:w1 + w] = arr[d_from:d_from + d, h_from:h_from + h, w_from:w_from + w]
        return arr_2