import os
import time
import pickle
import numpy as np
import torch
import adni_parcel_data
from fusion_model import FusionModel
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

fusion_data = './data/'
data_path = '../../../Data'
PreTrainPath = 'models/'

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

class FusionData(Dataset):
    def __init__(self, train = True):
        if train:
            self.data = pickle.load(open(fusion_data + 'train_data.pkl', "rb"))
            self.label = pickle.load(open(fusion_data + 'train_label.pkl', "rb"))
        else:
            self.data = pickle.load(open(fusion_data + 'test_data.pkl', "rb"))
            self.label = pickle.load(open(fusion_data + 'test_label.pkl', "rb"))

    def __getitem__(self, index):
        img = self.data[index]
        target = self.label[index]
        return img,target

    def __len__(self):
        return len(self.label)


def printData(data):
    import code
    print('*' * 50)
    print(len(data))
    for d, label in data:
        print(d[0].shape, label.shape)

    # 控制台交互，在命令行打印想要输出的结果，ctrl+D继续、exit()退出
    # code.interact(local=locals())

def calc_acc(pred, label):
    return (sum(pred == label) / len(pred))

def feature_fusion():
    Epoch = 500
    train_dataset = FusionData(train = True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # printData(train_loader)

    test_dataset = FusionData(train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # printData(test_loader)

    model = FusionModel().cuda()
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_func = torch.nn.BCEWithLogitsLoss().cuda()

    loss_point = []
    loss_point_test = []
    accuracy_point = []
    accuracy_point_test = []
    is_label_str = '_label'

    f = open(os.path.join('./result', 'result.txt'), 'a')

    for epoch in range(Epoch):
        model.train()
        sample_num = 0
        correct = 0
        loss_vec=[]
        for step, (batch_x, batch_y) in enumerate(train_loader):
            image = Variable(batch_x.cuda())
            out = model(image)
            idxs = [y for y in batch_y]
            target = Variable(torch.FloatTensor(idxs)).cuda()
            loss = loss_func(out.squeeze(1), target)
            loss_vec.append(loss.cpu().data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mask = out.cpu().ge(0.5).float()
            correct += sum([1.0 for i in range(batch_x.shape[0]) if mask.data.numpy()[i] == target.cpu().data.numpy()[i]])
            sample_num += batch_x.shape[0]
        print("Epoch [%d/%d], Acc: [%.4f]" % (epoch + 1, Epoch, correct / sample_num))
        loss_point.append(sum(loss_vec)/len(loss_vec))
        accuracy_point.append(correct / sample_num)
        torch.save(model.state_dict(), os.path.join('./result', 'FusionModel_' + str(epoch + 1) + '.pkl'))

        model.eval()
        test_correct = 0
        test_size = 0
        loss_vec_test=[]
        for step, (batch_x, batch_y) in enumerate(test_loader):
            idxs = [y for y in batch_y]
            target = Variable(torch.FloatTensor(idxs), volatile=True).cuda()
            image = Variable(batch_x, volatile=True).cuda()
            out = model(image)
            loss = loss_func(out.squeeze(1), target)
            loss_vec_test.append(loss.cpu().data[0])
            mask = out.cpu().ge(0.5).float()
            test_correct += sum(
                [1.0 for i in range(batch_x.shape[0]) if mask.data.numpy()[i] == target.cpu().data.numpy()[i]])
            test_size += batch_x.shape[0]
            if epoch == Epoch - 1:
                for i in range(batch_x.shape[0]):
                    f.write(str(target.cpu().data.numpy()[i]))
                    f.write('    ')
                    f.write(str(mask.data.numpy()[i]))
                    f.write('\n')

        print('test  Epoch: %d  Acc: %.4f' % (epoch + 1, test_correct / test_size))
        loss_point_test.append(sum(loss_vec_test)/len(loss_vec_test))
        accuracy_point_test.append(test_correct / test_size)
    f.close()
    model_dir = os.path.join('./result', 'Fusion.pkl')
    torch.save(model.state_dict(), model_dir)

    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    np.savetxt(os.path.join('./result', 'train_Fusion_loss'+str(time_str)+'.txt'), loss_point)
    np.savetxt(os.path.join('./result', 'test_Fusion_loss'+str(time_str)+'.txt'), loss_point_test)
    np.savetxt(os.path.join('./result', 'train_Fusion_acc'+str(time_str)+'.txt'), accuracy_point)
    np.savetxt(os.path.join('./result', 'test_Fusion_acc'+str(time_str)+'.txt'), accuracy_point_test)


def vote_fusion():
    train_pred = []
    train_label = []
    test_pred = []
    test_label = []
    for parcel_name, path in FC_Dict.items():
        model = torch.load(os.path.join(path))
        nDepth, nHeight, nWidth = parcel_label_name[parcel_name]
        # 重新计算符合AE模型的数据的输入尺寸（nDepth, nHeight, nWidth是全局变量，上面会用于计算全连接网络的输入的维度）
        s_range = np.array(parcel_label_name[parcel_name])
        reference_dim = np.array([i * 12 + 4 for i in range(10)])
        tmp = np.array([(i - 4) // 12 if (i - 4) % 12 == 0 else 1 + (i - 4) // 12 for i in s_range]) * 12 + 4
        nDepth, nHeight, nWidth = [int(i) for i in tmp]
        # train_data = adni_parcel_data.ADNI(is_train=True, is_label_input=False, parcel=parcel_name, is_autoencoder=True,
        #                              root_dir=data_path, transform=None)
        # train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=1)
        test_data = adni_parcel_data.ADNI(is_train=False, is_label_input=False, parcel=parcel_name, is_autoencoder=True,
                                     root_dir=data_path, transform=None)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=1)

        model.eval()
        test_correct = 0
        test_size = 0
        loss_vec_test = []
        loss_func = torch.nn.BCEWithLogitsLoss().cuda()

        temp_pred = []
        temp_label = []

        for step, (batch_x, batch_y, filename) in enumerate(test_loader):
            idxs = [y for y in batch_y]
            target = Variable(torch.FloatTensor(idxs), volatile=True).cuda()
            image = Variable(batch_x.view(-1, 1, nDepth, nHeight, nWidth), volatile=True).cuda()
            out = model(image)
            loss = loss_func(out.squeeze(1), target)
            loss_vec_test.append(loss.cpu().data[0])
            mask = out.cpu().ge(0.5).float()
            test_correct += sum(
                [1.0 for i in range(batch_x.shape[0]) if mask.data.numpy()[i] == target.cpu().data.numpy()[i]])
            test_size += batch_x.shape[0]


            temp_pred.append(mask.data.numpy())
            temp_label.append(target.cpu().data.numpy())

        test_pred.append(np.squeeze(np.array(temp_pred)))
        test_label.append(np.squeeze(np.array(temp_label)))

    ensemble_pred = np.mean(test_pred, axis=0) > 0 #投票，设置成0.5（少数服从多数）比设置成0（提名）效果要差
    ensemble = [1.0 if x else 0.0 for x in ensemble_pred]

    # print(ensemble)
    # print('\n', list(test_label[0]))

    print()
    for i, (parcel_name, path) in enumerate(FC_Dict.items()):
        print(parcel_name+':', calc_acc(test_pred[i], test_label[0]))

    print('Ensemble:', calc_acc(ensemble, test_label[0]))


if __name__ == '__main__':

    feature_fusion() #特征级融合

    vote_fusion()  # 决策级融合
