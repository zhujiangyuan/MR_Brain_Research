import os
import numpy as np
import torch
import fusion_model
import  pickle
import adni_parcel_data
from torch.autograd import Variable

data_path = '../../../Data'
PreTrainPath = 'models/'
Dict = {
    'Amygdala': PreTrainPath + 'ACEModel_Amygdala8825_not_label.pkl',
    'CaudateNucleus': PreTrainPath + 'ACEModel_CaudateNucleus8825_not_label.pkl',
    'EntorhinalArea': PreTrainPath + 'ACEModel_EntorhinalArea8825_not_label.pkl',
    'GlobusPallidu': PreTrainPath + 'ACEModel_GlobusPallidu8825_not_label.pkl',
    'Hippocampus': PreTrainPath + 'ACEModel_Hippocampus8825_not_label.pkl',
    'Insular': PreTrainPath + 'ACEModel_Insular8825_not_label.pkl',
    'Putamen': PreTrainPath + 'ACEModel_Putamen8825_not_label.pkl',
    'Thalamus': PreTrainPath + 'ACEModel_Thalamus8825_not_label.pkl'
}

FC_Dict = {
    'Amygdala': PreTrainPath + 'ACE_FullConnectModel_Amygdala8835_not_label.pkl',
    'CaudateNucleus': PreTrainPath + 'ACE_FullConnectModel_CaudateNucleus8835_not_label.pkl',
    'EntorhinalArea': PreTrainPath + 'ACE_FullConnectModel_EntorhinalArea8835_not_label.pkl',
    'GlobusPallidu': PreTrainPath + 'ACE_FullConnectModel_GlobusPallidu8835_not_label.pkl',
    'Hippocampus': PreTrainPath + 'ACE_FullConnectModel_Hippocampus8835_not_label.pkl',
    'Insular': PreTrainPath + 'ACE_FullConnectModel_Insular8835_not_label.pkl',
    'Putamen': PreTrainPath + 'ACE_FullConnectModel_Putamen8835_not_label.pkl',
    'Thalamus': PreTrainPath + 'ACE_FullConnectModel_Thalamus8835_not_label.pkl'
}

parcel_label_name = {
    'Putamen': [70, 39, 54],
    'Insular': [83, 56, 68],
    'Amygdala': [75, 33, 33],
    'Thalamus': [57, 49, 47],
    'Hippocampus': [88, 45, 63],
    'GlobusPallidu': [66, 31, 38],
    'CaudateNucleus': [64, 77, 107],
    'EntorhinalArea': [68, 32, 32]
}

def saveData(train = True):
    '''
    根据训练好的Encoder模型处理原始数据，将编码后的特征拼接后保存
    :param train: bool, 保存训练或测试数据
    :return: None
    '''
    features = []
    labels = []
    for parcel_name, path in Dict.items():
        temp = []
        target = []
        pre_model = torch.load(os.path.join(path))
        nDepth, nHeight, nWidth = parcel_label_name[parcel_name]
        # 重新计算符合AE模型的数据的输入尺寸（nDepth, nHeight, nWidth是全局变量，上面会用于计算全连接网络的输入的维度）
        s_range = np.array(parcel_label_name[parcel_name])
        reference_dim = np.array([i * 12 + 4 for i in range(10)])
        tmp = np.array([(i - 4) // 12 if (i - 4) % 12 == 0 else 1 + (i - 4) // 12 for i in s_range]) * 12 + 4
        nDepth, nHeight, nWidth = [int(i) for i in tmp]
        if train:
            data = adni_parcel_data.ADNI(is_train=True, is_label_input=False, parcel=parcel_name, is_autoencoder=True,
                                         root_dir=data_path, transform=None)
            # loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=1)
        else:
            data = adni_parcel_data.ADNI(is_train=False, is_label_input=False, parcel=parcel_name, is_autoencoder=True,
                                         root_dir=data_path, transform=None)
            # loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=128, shuffle=False, num_workers=1)

        # for step, (batch_x, batch_label, filename) in enumerate(loader):
        #     image = Variable(batch_x.view(-1, 1, nDepth, nHeight, nWidth).cuda(), volatile=True)
        #     label = Variable(batch_x.view(-1, nDepth, nHeight, nWidth).cuda(), volatile=True)
        #
        #     print(image.shape)
        #
        #     pre_model.eval()
        #     pre_model.cuda()
        #
        #     _, output = pre_model(image)
        #
        #     loss_func = torch.nn.MSELoss(size_average=True).cuda()
        #     loss = loss_func(label, output)
        #     print(loss)


        for step, (batch_x, batch_label, filename) in enumerate(data):
            image = Variable(batch_x.view(-1, 1, nDepth, nHeight, nWidth).cuda(), volatile=True)
            # label = Variable(batch_x.view(-1, nDepth, nHeight, nWidth).cuda(), volatile=True)
            pre_model.eval()
            pre_model.cuda()

            feature, _ = pre_model(image)

            temp.append(feature.view(16, -1).cpu().data.numpy())
            target.append(batch_label)

        print(np.array(temp).shape)
        features.append(np.array(temp))
        labels.append(target)

    features = np.concatenate(tuple(features), axis=2)
    print(features.shape)
    # print(labels[0])
    # print(labels[1])
    # print(labels[2])
    # print(np.array(labels).shape)
    if train:
        data = open('data/train_data.pkl', 'wb')
        label = open('data/train_label.pkl', 'wb')
    else:
        data = open('data/test_data.pkl', 'wb')
        label = open('data/test_label.pkl', 'wb')

    # Pickle dictionary using protocol 0.
    pickle.dump(features, data)
    pickle.dump(labels[0], label)
    data.close()
    label.close()


if __name__ == '__main__':
    saveData(True)
    saveData(False)