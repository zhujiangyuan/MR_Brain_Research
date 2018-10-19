#-*- coding:utf-8 -*-
import argparse
import torch
import os
import math
import numpy as np
from torch.autograd import Variable
import time
import shutil

import adni_model
import adni_parcel_data

global nDepth, nHeight, nWidth
parcel_label_name = {
    'Putamen': [70, 39, 54],
    'Insular': [83, 56, 68],
    # # 'Cingulum': [66, 83, 108], #会使程序中断，暂时不用
    'Amygdala': [75, 33, 33],
    'Thalamus': [57, 49, 47],
    # 'LateralVentricle': [125, 183, 161],
    'Hippocampus': [88, 45, 63],
    'GlobusPallidu': [66, 31, 38],
    #'NucleusAccumbens': [48, 24, 31],
    'CaudateNucleus': [64, 77, 107],
    'EntorhinalArea': [68, 32, 32]
}

def trainAEModel(args, parcel_name):
    global nDepth, nHeight, nWidth
    if not args.if_AE:
        print('not use AE to extract features!!')
        return
    if args.des_in_dir != '' and os.path.exists(os.path.join(args.des_in_dir, args.AEModelName)):
        if not os.path.exists(os.path.join(args.des_out_dir, args.AEModelName)):
            shutil.copy2(os.path.join(args.des_in_dir, args.AEModelName), os.path.join(args.des_out_dir, args.AEModelName))
        print('AE Model already exist, the same path and name!!')
        return

    train_data = adni_parcel_data.ADNI(is_train=True, is_label_input=args.if_label, parcel=parcel_name, is_autoencoder=args.if_AE, root_dir=args.data_dir, transform=None)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.train1, shuffle=True, num_workers=1)
    test_data = adni_parcel_data.ADNI(is_train=False, is_label_input=args.if_label, parcel=parcel_name, is_autoencoder=args.if_AE, root_dir=args.data_dir, transform=None)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.test1, shuffle=False, num_workers=1)

    torch.cuda.manual_seed(1)
    cnn_AE_model = adni_model.CNNAutoEncoder()

    cnn_AE_model.cuda()
    loss_func = torch.nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(cnn_AE_model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.AEMidModelName != '' and args.des_in_dir != '':
        cnn_AE_model.load_state_dict(torch.load(os.path.join(args.des_in_dir, args.AEMidModelName)))
        start_epoch = int(args.AEMidModelName.split('.')[0].split('_')[-1])

    loss_point = []
    loss_point_test = []

    for epoch in range(start_epoch, args.epoch1):
        cnn_AE_model.train()
        train_loss = []
        for step, (batch_x, batch_label, filename) in enumerate(train_loader):
            image = Variable(batch_x.view(-1, 1, nDepth, nHeight, nWidth).cuda())
            label = Variable(batch_x.view(-1, nDepth, nHeight, nWidth).cuda())  #label=image  autoencoder
            encoder_features, outputs = cnn_AE_model(image)
            # print(image.shape, encoder_features.shape, outputs.shape)
            # input('******************************')

            loss = loss_func(outputs, label)    # mean square error

            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients

            train_loss.append(loss.cpu().data[0])

            if step % 20 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f  avg_Loss: %.4f'   #1000* just for show, no other reason
                    %(epoch+1, args.epoch1, step+1, len(train_data)//args.train1, 1000*loss.data[0], 1000*(sum(train_loss) / len(train_loss))))
        loss_point.append(sum(train_loss) / len(train_loss))
        torch.save(cnn_AE_model.state_dict(), os.path.join(args.des_out_dir, 'trainAEModel_' + parcel_name +'_'+ str(epoch+1) + '.pkl'))

        cnn_AE_model.eval()  # test pattern
        test_loss = []
        for step, (batch_x, batch_label, filename) in enumerate(test_loader):
            image = Variable(batch_x.view(-1, 1, nDepth, nHeight, nWidth).cuda())
            label = Variable(batch_x.view(-1, nDepth, nHeight, nWidth).cuda())

            _, outputs = cnn_AE_model(image)
            loss = loss_func(outputs, label)
            test_loss.append(loss.cpu().data[0])

        print('Epoch: %d  avg_Loss: %.4f'%(epoch+1, 1000*(sum(test_loss) / len(test_loss))))
        loss_point_test.append(sum(test_loss)/len(test_loss))

    model_dir = os.path.join(args.des_out_dir, args.AEModelName)
    torch.save(cnn_AE_model.state_dict(), model_dir)   #保存模型参数
    torch.save(cnn_AE_model, os.path.join('./models', args.AEModelName)) # 保存完整模型
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    np.savetxt(os.path.join(args.des_out_dir, 'train_AE_loss'+ parcel_name + '_'+str(time_str)+'.txt'), loss_point)
    np.savetxt(os.path.join(args.des_out_dir, 'test_AE_loss'+ parcel_name + '_'+str(time_str)+'.txt'), loss_point_test)


def trainFullConnectModel(args, parcel_name):
    global nDepth, nHeight, nWidth
    train_data = adni_parcel_data.ADNI(is_train=True, is_label_input=args.if_label, parcel=parcel_name, is_autoencoder=args.if_AE, root_dir=args.data_dir, transform=None)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.train2, shuffle=True, num_workers=1)
    test_data = adni_parcel_data.ADNI(is_train=False, is_label_input=args.if_label, parcel=parcel_name, is_autoencoder=args.if_AE, root_dir=args.data_dir, transform=None)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.test2, shuffle=True, num_workers=1)

    feature_num = 8*(((nDepth-4)//12)-2)*(((nHeight-4)//12)-2)*(((nWidth-4)//12)-2)

    model = adni_model.totalModel(feature_num, 128).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_func = torch.nn.BCEWithLogitsLoss().cuda()

    if args.if_AE and os.path.exists(os.path.join(args.des_out_dir, args.AEModelName)):
        #如何导入预训练模型
        cnn_AE_model = adni_model.CNNAutoEncoder().cuda()
        cnn_AE_model.load_state_dict(torch.load(os.path.join(args.des_out_dir, args.AEModelName)))
        pretrained_dict = cnn_AE_model.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(model)
        for para in list(model.parameters())[:6]:
            para.requires_grad = False
        #只训练全连接的部分
        optimizer = torch.optim.Adam([{'params': model.cov1.parameters()},
                                      {'params': model.fc1.parameters()},
                                      {'params': model.bn_fc.parameters()},
                                      {'params': model.fc2.parameters()}], lr=args.lr, weight_decay=1e-5)

    print('start to train total model...')

    loss_point = []
    loss_point_test = []
    accuracy_point = []
    accuracy_point_test = []
    is_label_str = '_label'
    if not args.if_label:
        is_label_str = '_not_label'
    f = open(os.path.join(args.des_out_dir, parcel_name+str(is_label_str) + '.txt'), 'a')

    for epoch in range(args.epoch2):
        model.train()
        sample_num = 0
        correct = 0
        loss_vec=[]
        for step, (batch_x, batch_y, filename) in enumerate(train_loader):
            image = Variable(batch_x.view(-1, 1, nDepth, nHeight, nWidth).cuda())
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
        print("Epoch [%d/%d], Acc: [%.4f]" % (epoch + 1, args.epoch2, correct / sample_num))
        loss_point.append(sum(loss_vec)/len(loss_vec))
        accuracy_point.append(correct / sample_num)
        torch.save(model.state_dict(), os.path.join(args.des_out_dir, 'trainFullConnectModel_' + parcel_name +'_'+ str(epoch + 1) + '.pkl'))

        model.eval()
        test_correct = 0
        test_size = 0
        loss_vec_test=[]
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
            if epoch == args.epoch2 - 1:
                for i in range(batch_x.shape[0]):
                    f.write(str(target.cpu().data.numpy()[i]))
                    f.write('    ')
                    f.write(str(mask.data.numpy()[i]))
                    f.write('\n')

        print('test  Epoch: %d  Acc: %.4f' % (epoch + 1, test_correct / test_size))
        loss_point_test.append(sum(loss_vec_test)/len(loss_vec_test))
        accuracy_point_test.append(test_correct / test_size)
    f.close()
    model_dir = os.path.join(args.des_out_dir, args.TotalModelName)
    torch.save(model.state_dict(), model_dir)
    torch.save(model, os.path.join('./models', args.AEModelName))  # 保存完整模型

    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    np.savetxt(os.path.join(args.des_out_dir, 'train_FullConnect_loss'+ parcel_name + '_'+str(time_str)+'.txt'), loss_point)
    np.savetxt(os.path.join(args.des_out_dir, 'test_FullConnect_loss'+ parcel_name + '_'+str(time_str)+'.txt'), loss_point_test)
    np.savetxt(os.path.join(args.des_out_dir, 'train_FullConnect_acc'+ parcel_name + '_'+str(time_str)+'.txt'), accuracy_point)
    np.savetxt(os.path.join(args.des_out_dir, 'test_FullConnect_acc'+ parcel_name + '_'+str(time_str)+'.txt'), accuracy_point_test)

def trainProcess(args, parcel_name):
    trainAEModel(args, parcel_name)
    trainFullConnectModel(args, parcel_name)

def getParameters():
    args = argparse.ArgumentParser(description='Fourth model', epilog='Adjust some parameters')
    # necessary
    args.add_argument('--name1', type=str, default='', help='AE model name', dest = 'AEModelName')

    args.add_argument('--name2', type=str, default='', help='Total model name', dest = 'TotalModelName')

    args.add_argument("--middle_name1", type=str, help="AE model name which want to be continued", default='', dest='AEMidModelName')

    args.add_argument('--minpath', type=str, default=r'../142_file/train_record', help='the path of the pretrained AE model', dest='des_in_dir')

    args.add_argument('--moutpath', type=str, default=r'../142_file/train_record', help='the path to save model', dest = 'des_out_dir')

    args.add_argument('--dpath', type=str, default=r'../../../Data', help='the path of datasets', dest = 'data_dir')

    args.add_argument("--m", type=float, help="momentum", default=0.90, dest = 'momentum')

    args.add_argument("--epoch1", type=int, help="AE training epoch", default=25, dest = 'epoch1')

    args.add_argument("--train1", type=int, help="AE training batch size", default=8, dest = 'train1')

    args.add_argument("--test1", type=int, help="AE testing batch size", default=8, dest = 'test1')

    args.add_argument("--epoch2", type=int, help="Classifier training epoch", default=35, dest = 'epoch2')

    args.add_argument("--train2", type=int, help="Classifier training batch size", default=8, dest = 'train2')

    args.add_argument("--test2", type=int, help="Classifier testing batch size", default=8, dest = 'test2')

    args.add_argument("--lr", type=float, dest="lr", help='learning rate', default=1e-3)

    args.add_argument('-AE', action='store_true', default=True, help="if use AE to extract the feature (pretrain the total model)", dest='if_AE')
    args.add_argument('-CNN', action='store_false', default=True, help="if not use AE, use only CNN", dest='if_AE')

    args.add_argument('-label', help="input label or voxel value of dicom", dest='if_label', action='store_true', default=False)
    args.add_argument('-not_label', help="input label or voxel value of dicom", dest='if_label', action='store_false', default=False)

    args = args.parse_args()
    return args

if __name__ == '__main__':
    global nDepth, nHeight, nWidth
    args = getParameters()

    is_label_str = '_label'
    if not args.if_label:
        print('not label')
        is_label_str = '_not_label'
    else:
        print('is_label')

    if not os.path.isdir(args.des_out_dir):
        os.mkdir(args.des_out_dir)

    for parcel_name in parcel_label_name:
        print(parcel_name)
        nDepth, nHeight, nWidth = parcel_label_name[parcel_name]
        if args.if_AE:
            #重新计算符合AE模型的数据的输入尺寸（nDepth, nHeight, nWidth是全局变量，上面会用于计算全连接网络的输入的维度）
            s_range = np.array(parcel_label_name[parcel_name])
            reference_dim = np.array([i*12+4 for i in range(10)])
            tmp = np.array([(i-4)//12 if (i-4)%12==0 else 1+(i-4)//12 for i in s_range])*12 + 4
            nDepth, nHeight, nWidth = [int(i) for i in tmp]
        args.AEModelName = 'ACEModel_' + parcel_name + str(args.train1) + str(args.test1) + str(args.epoch1) + is_label_str + '.pkl'
        args.TotalModelName = 'ACE_FullConnectModel_' + parcel_name + str(args.train2) + str(args.test2) + str(args.epoch2) + is_label_str + '.pkl'
        print(args)

        trainProcess(args, parcel_name)