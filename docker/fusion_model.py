import os
import torch
import math

class FusionModel(torch.nn.Module):
    '''
    融合模型的的定义，用于根据拼接后的特征得出分类结果
    模型的输入是多通道的一维特征数据，是二维数据
    '''
    def __init__(self):
        super(FusionModel, self).__init__()
        self.encoder = torch.nn.Sequential(                       #16 *274
            torch.nn.Conv1d(16, 16, 7, stride=1),                 #16 *268
            # torch.nn.BatchNorm1d(16),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(True),
            torch.nn.MaxPool1d(2, stride=2),                      #16 *134
            torch.nn.Conv1d(16, 16, 5, stride=2),                 #16 *65
            torch.nn.Dropout(0.5),
            # torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(True),
            torch.nn.MaxPool1d(2, stride=2),                      #16 *32
            torch.nn.Conv1d(16, 32, 5, stride=2),                 #16 *14
            # torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(True),
        )

        for m in self.modules():
            self.weights_init(m)

        self.fc1 = torch.nn.Linear(32*14, 32)
        self.dropout1 = torch.nn.Dropout(0.5)
        # self.bn_fc = torch.nn.BatchNorm1d(32)
        self.relu1 = torch.nn.ReLU(True)
        self.fc2 = torch.nn.Linear(32, 1)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.xavier_normal(m.weight.data, gain=math.sqrt(2.0))

        if classname.find('Linear') != -1:
            torch.nn.init.xavier_normal(m.weight.data, gain=math.sqrt(2.0))

    def forward(self, x):
        encoder_feature = self.encoder(x)
        encoder_feature = encoder_feature.view(encoder_feature.size(0), -1)
        # print(encoder_feature.shape)
        out = self.fc1(encoder_feature)
        out = self.dropout1(out)
        # out = self.bn_fc(out)
        out = self.relu1(out)
        pred = self.fc2(out)
        return pred
