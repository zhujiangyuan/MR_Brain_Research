import torch.utils.data as data
import os
import os.path
import numpy as np
import torch
import dicom

np.set_printoptions(threshold=np.inf)

nc_train = 969
ad_train = 535
nc_test = 245
ad_test = 135

#对于parcel的处理方法：
#1、将原图扩充到一个统一尺寸后，如184*256*256
# 对training set中的样本取得每一个parcel在x,y,z方向上的范围
# 然后对training set中的所有样本取并集，之后经过适当调整以满足模型输入的尺寸需要
#[zmin, zmax, xmin, xmax, ymin, ymax] considering the AutoEncoder_cnn model

#2、图像是否扩充不作要求
# 将所有样本的指定parcel的部分抠出，抠出的parcel有大小
# 然后对training set中所有样本取并集，之后经过适当调整适应模型需要，对每个抠出的parcel进行边缘填充0的扩充。
#理论上这种方法应该可以部分去除未经过配准操作导致的parcel位置上的差异对结果的影响。
#[xrange,yrange,zrange]
#如果用tofile保存的parcel信息，那么需要在保存parcel的时候就已经做好扩充，因为fromfile读到的数据是一维的没有任何格式信息，需要自己格式化。
parcels_range = {
    'Putamen': [70, 39, 54],
    'Insular': [83, 56, 68],
    'Cingulum': [66, 83, 108],
    'Amygdala': [75, 33, 33],
    'Thalamus': [57, 49, 47],
    'LateralVentricle': [125, 183, 161],
    'Hippocampus': [88, 45, 63],
    'GlobusPallidu': [66, 31, 38],
    'NucleusAccumbens': [48, 24, 31],
    'CaudateNucleus': [64, 77, 107],
    'EntorhinalArea': [68, 32, 32]
}

parcel_map = {
    'Putamen' : 'PutamenParcel.dat',
    'Insular' : 'InsularParcel.dat',
    'Cingulum' : 'CingulumParcle.dat',
    'Amygdala' : 'AmygdalaParcel.dat',
    'Thalamus' :'ThalamusParcel.dat',
    'LateralVentricle' : 'VentricleParcel.dat',
    'Hippocampus' : 'HippocampusParcel.dat',
    'GlobusPallidu' : 'GlobusPalliduParcel.dat',
    'EntorhinalArea':'EntorhinalAreaParcel.dat',
    'CaudateNucleus' : 'CaudateNucleusParcel.dat',
    'NucleusAccumbens' :'NucleusAccumbensParcel.dat',
}

#该文件使用方法2
class ADNI(data.Dataset):

    def __init__(self, is_train=True, is_label_input=False, is_autoencoder=True, parcel='Hippocampus', root_dir=r'/AD_NC_data', transform=None, target_transform=None):
        self.transform = transform
        self.target_transform=target_transform
        self.is_train = is_train
        self.is_label_input = is_label_input
        self.is_AE = is_autoencoder
        self.parcel = parcel
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

        target = self.labels[index]
        image = self.get_each_raw_data(index)
        image = self.TmpNormalization(image)
        image = torch.from_numpy(image)


        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target, self.data[index]

    def __len__(self):
        return len(self.data)


    def TmpNormalization(self, x):
        if self.is_label_input:#只使用掩模进行训练
            tmp = x
            tmp[tmp > 0] = 1
            tmp = np.array(tmp, dtype=np.float32)
        else:#使用Dicom原始值进行训练
            # 2018.09.07修改归一化方法
            tmp = (1.0*x-x.min())/(x.max()-x.min())
            tmp = np.array(tmp, dtype=np.float32)
        return tmp

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
        NC_parcel_dir = os.path.join(self.root_dir, 'NC_parcel')
        AD_parcel_dir = os.path.join(self.root_dir, 'AD_parcel')
        if self.labels[index] == 0:
            image = np.fromfile(os.path.join(NC_parcel_dir, filename, parcel_map[self.parcel]), dtype=np.uint16)
        else:
            image = np.fromfile(os.path.join(AD_parcel_dir, filename, parcel_map[self.parcel]), dtype=np.uint16)
        image.shape = parcels_range[self.parcel]
        if self.is_AE:
            #重新计算符合AE模型的数据输入
            s_range = np.array(parcels_range[self.parcel])
            ndepth, nheight, nwidth = np.array(
                [(i - 4) // 12 if (i - 4) % 12 == 0 else 1 + (i - 4) // 12 for i in s_range]) * 12 + 4
            return self.add_boundingbox(np.array(image), ndepth, nheight, nwidth)
        else:
            return image

    def add_boundingbox(self, arr, depth_range, height_range, width_range):
        #根据要求的depth, height, width调整图像
        arr_2 = np.zeros((depth_range, height_range, width_range), dtype=np.float32)
        d, h, w = arr.shape
        d1 = depth_range//2 - d // 2
        h1 = height_range//2 - h // 2
        w1 = width_range//2 - w // 2

        arr_2[d1:d1 + d, h1:h1 + h, w1:w1 + w] = arr[:, :, :]
        return arr_2