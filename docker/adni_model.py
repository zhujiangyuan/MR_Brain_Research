import torch
import math

class CNNAutoEncoder(torch.nn.Module):
    def __init__(self):
        super(CNNAutoEncoder, self).__init__()
        #https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
        self.encoder = torch.nn.Sequential(                             #1* 184*256*256
            torch.nn.Conv3d(1, 16, 3, stride=3, padding=1), #16* 62*86*86
            torch.nn.BatchNorm3d(16),
            # torch.nn.ReLU(True),
            torch.nn.MaxPool3d(2, stride=2),                #16* 31*43*43

            torch.nn.Conv3d(16, 16, 3, stride=2, padding=1), #8* 16*22*22
            torch.nn.BatchNorm3d(16),
            # torch.nn.ReLU(True),
            torch.nn.MaxPool3d(2, stride=1)                 #8* 15*21*21
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(16, 16, 3, stride=2),   #16* 31*43*43
            torch.nn.BatchNorm3d(16),
            # torch.nn.ReLU(True),
            torch.nn.ConvTranspose3d(16, 8, 5, stride=3, padding=1),#8* 93*129*129
            torch.nn.BatchNorm3d(8),
            # torch.nn.ReLU(True),
            torch.nn.ConvTranspose3d(8, 1, 2, stride=2, padding=1), #1* 184*256*256
            torch.nn.BatchNorm3d(1),
            torch.nn.Tanh()
        )

        for m in self.modules():
            self.weights_init(m)

    def forward(self, x):
        encoder_feature = self.encoder(x)
        decoder_result = self.decoder(encoder_feature)
        return encoder_feature, decoder_result

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv')!=-1:
            torch.nn.init.xavier_normal(m.weight.data, gain=math.sqrt(2.0))  #can also use 'torch.nn.init.kaiming_normal'

class totalModel(torch.nn.Module):
    def __init__(self, D_in, H):
        super(totalModel, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv3d(1, 16, 3, stride=3, padding=1), #16* 62*86*86
            torch.nn.BatchNorm3d(16),
            # torch.nn.ReLU(True),
            torch.nn.MaxPool3d(2, stride=2),                #16* 31*43*43
            torch.nn.Conv3d(16, 16, 3, stride=2, padding=1), #8* 16*22*22
            torch.nn.BatchNorm3d(16),
            # torch.nn.ReLU(True),
            torch.nn.MaxPool3d(2, stride=1)               #8* 15*21*21
        )

        self.cov1 = torch.nn.Conv3d(16, 8, 3, stride=1)
        self.fc1 = torch.nn.Linear(D_in, H)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.bn_fc = torch.nn.BatchNorm1d(H)
        self.relu1 = torch.nn.ReLU(True)
        self.fc2 = torch.nn.Linear(H, 1)

        torch.nn.init.xavier_normal(self.fc1.weight, gain=math.sqrt(2.0))
        torch.nn.init.xavier_normal(self.fc2.weight, gain=1)

    def forward(self, x):
        encoder_feature = self.encoder(x)            # 将编码后的特征拉伸，获取特征用于训练，可能会损失结构信息
        encoder_feature = self.cov1(encoder_feature) # 20180914添加需要训练的卷积
        encoder_feature = encoder_feature.view(encoder_feature.size(0), -1)
        out = self.fc1(encoder_feature)
        out = self.dropout1(out)
        out = self.bn_fc(out)
        out = self.relu1(out)
        pred = self.fc2(out)
        return pred

