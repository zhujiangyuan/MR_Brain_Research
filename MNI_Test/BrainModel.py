import torch


class BrainModel(torch.nn.Module):
    def __init__(self, D_in = 156*157*189, H=2):
        super(BrainModel, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv3d(1, 16, 3, stride=3, padding=1), #16* 62*86*86
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(True),
            torch.nn.MaxPool3d(2, stride=2),             #16* 31*43*43
            torch.nn.Conv3d(16, 8, 3, stride=3, padding=1), #16* 62*86*86
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU(True),
            torch.nn.MaxPool3d(2, stride=2),      
        )

        self.cov1 = torch.nn.Conv3d(8, 8, 3, stride=1)
        self.fc1 = torch.nn.Linear(D_in, 8)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.bn_fc = torch.nn.BatchNorm1d(1)
        self.relu1 = torch.nn.ReLU(True)
        self.fc2 = torch.nn.Linear(8, 4)
        self.fc3 = torch.nn.Linear(4, 2)
        self.fc4 = torch.nn.Linear(2, 1)
    def forward(self, x):
        #encoder_feature = self.encoder(x)            # 将编码后的特征拉伸，获取特征用于训练，可能会损失结构信息
    
        #encoder_feature = self.cov1(encoder_feature) # 20180914添加需要训练的卷积
       
        #encoder_feature_size=encoder_feature.size(0)
        encoder_feature = x.view(-1, 156*157*189)
        out = self.fc1(encoder_feature)
        out = self.dropout1(out)
        #out = self.bn_fc(out)
        out = self.relu1(out)
        pred = self.fc2(out)
        pred = self.relu1(pred)
        pred = self.fc3(pred)
        pred = self.relu1(pred)
        pred = self.fc4(pred)
        return pred