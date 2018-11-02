import torch
import math


class CNNModel(torch.nn.Module):
    def __init__(self, D_in = 576, H = 100):
        super(CNNModel, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv3d(1, 16, 3, stride=3, padding=1), #16* 62*86*86
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(True),
            torch.nn.MaxPool3d(2, stride=2),                #16* 31*43*43
            torch.nn.Conv3d(16, 8, 3, stride=2, padding=1), #8* 16*22*22
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU(True),
            torch.nn.MaxPool3d(2, stride=1)                 #8* 15*21*21
        )

        self.fc1 = torch.nn.Linear(D_in, H)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.bn_fc = torch.nn.BatchNorm1d(H)
        self.relu1 = torch.nn.ReLU(True)
        self.out = torch.nn.Linear(H, 1)

        torch.nn.init.xavier_normal(self.fc1.weight, gain=math.sqrt(2.0))
        torch.nn.init.xavier_normal(self.out.weight, gain=1)

    def forward(self, x):
        encoder_feature = self.encoder(x)
        encoder_feature = encoder_feature.view(encoder_feature.size(0), -1)
        out = self.fc1(encoder_feature)
        out = self.bn_fc(out)
        out = self.relu1(out)
        pred = self.out(out)
        #pred = self.out(self.relu1(self.bn_fc(self.fc1(encoder_feature))))
        return pred