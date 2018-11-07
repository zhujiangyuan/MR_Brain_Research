import torch
import DataSet
import CNNModel
import datetime 
import os
from torch.autograd import Variable

if __name__ == '__main__':

    log_path = r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\log'
    log_file_name = 'Train_log_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt'
    log_full_Path = os.path.join(log_path,log_file_name)
    fp = open(log_full_Path + log_file_name, 'w')
    train_data = DataSet.DataSet(r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\Test_Case_Parcel_Range\TestCase_Hippocampus\Train')
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=10, shuffle=True, num_workers=1)
    test_data = DataSet.DataSet(r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\Test_Case_Parcel_Range\TestCase_Hippocampus\Test')
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=1)
    model = CNNModel.CNNModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-1)
    loss_func = torch.nn.BCEWithLogitsLoss().cuda()
    
    optimizer.zero_grad()
    count = 0
    for epoch in range(80):
        model.train()
        loss_vec = []
        sample_num = 0
        correct = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            count += 1
            print(count)
            height,width,depth = train_data.getRawDataDimension()
            image = Variable(batch_x.view(-1, 1, depth, height, width).cuda())
            out = model(image)
            idxs = [y for y in batch_y]
            target = Variable(torch.FloatTensor(idxs)).cuda()
            loss = loss_func(out.squeeze(1), target)
            loss.cpu()
            loss_data = loss.cpu().data
            print(loss_data)
            loss_vec.append(loss.cpu().data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mask = out.cpu().ge(0.5).float()
            correct += sum([1.0 for i in range(batch_x.shape[0]) if mask.data.numpy()[i] == target.cpu().data.numpy()[i]])
            sample_num += batch_x.shape[0]
            #print("Epoch [%d/%d], Acc: [%.4f]" % (epoch + 1, args.epoch2, correct / sample_num))
            #loss_point.append(sum(loss_vec)/len(loss_vec))
            #accuracy_point.append(correct / sample_num)
        print('****Train Acc:' + str(correct / sample_num) + 'Train Loss' + str(loss_data[0]))
        fp.write('****Epoch_' + str(epoch) + '****Train Acc:' + str(correct / sample_num) + 'Train Loss' + str(loss_data[0]) + '\n')
            #torch.save(model.state_dict(), os.path.join(args.des_out_dir, 'trainFullConnectModel_' + parcel_name +'_'+ str(epoch + 1) + '.pkl'))
        model.eval()
        test_correct = 0
        test_size = 0
        loss_vec_test=[]
        for step, (batch_x, batch_y) in enumerate(test_loader):
            idxs = [y for y in batch_y]
            target = Variable(torch.FloatTensor(idxs), volatile=True).cuda()
            height,width,depth = test_data.getRawDataDimension()
            image = Variable(batch_x.view(-1, 1, depth, height, width).cuda())
            out = model(image)
            loss = loss_func(out.squeeze(1), target)
            loss_vec_test.append(loss.cpu().data[0])
            mask = out.cpu().ge(0.5).float()
            test_correct += sum(
                    [1.0 for i in range(batch_x.shape[0]) if mask.data.numpy()[i] == target.cpu().data.numpy()[i]])
            test_size += batch_x.shape[0]
        print('****Test Acc:' + str(test_correct / test_size) + 'Test Loss' + str(loss_data[0]))
        fp.write('****Epoch_' + str(epoch) + '****Test Acc:' + str(test_correct / test_size) + 'Test Loss' + str(loss_data[0]) + '\n')
    fp.close()      
print('finish')