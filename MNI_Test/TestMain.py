import torch
import DataSet
import CNNModel
from torch.autograd import Variable
if __name__ == '__main__':




    
    train_data = DataSet.DataSet(r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\Test_Case_Parcel_Range\TestCase_Putamen\Train')
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=10, shuffle=True, num_workers=1)

    model = CNNModel.CNNModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-1)
    loss_func = torch.nn.BCEWithLogitsLoss().cuda()
    loss_vec = []
    sample_num = 0
    correct = 0
    optimizer.zero_grad()
    for epoch in range(100):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            height,width,depth = train_data.getRawDataDimension()
            image = Variable(batch_x.view(-1, 1, depth, height, width).cuda())
            out = model(image)
            idxs = [y for y in batch_y]
            target = Variable(torch.FloatTensor(idxs)).cuda()
            loss = loss_func(out.squeeze(1), target)
            loss.cpu()
            loss_data = loss.cpu().data
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
            print(correct / sample_num)
            #torch.save(model.state_dict(), os.path.join(args.des_out_dir, 'trainFullConnectModel_' + parcel_name +'_'+ str(epoch + 1) + '.pkl'))

print('finish')