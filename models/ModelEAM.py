import torch
import torch.nn as nn
import torch.nn.functional as F

class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=4, dilation=4)
        
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.conv11 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv1 = F.relu(self.conv2(conv1))
        
        conv2 = F.relu(self.conv3(x))
        conv2 = F.relu(self.conv4(conv2))
        
        concat = torch.cat([conv1, conv2], dim=1)
        conv3 = F.relu(self.conv5(concat))
        add1 = x + conv3
        
        conv4 = F.relu(self.conv6(add1))
        conv4 = self.conv7(conv4)
        add2 = conv4 + add1
        add2 = F.relu(add2)
        
        conv5 = F.relu(self.conv8(add2))
        conv5 = F.relu(self.conv9(conv5))
        conv5 = self.conv10(conv5)
        add3 = add2 + conv5
        add3 = F.relu(add3)
        
        gap = self.gap(add3)
        gap = gap.view(gap.size(0), 64, 1, 1)
        conv6 = F.relu(self.conv11(gap))
        conv6 = torch.sigmoid(self.conv12(conv6))
        
        mul = conv6 * add3
        out = x + mul  # This is not included in the reference code
        return out

class RIDNet(nn.Module): #Use 230 for FINCH, Use 220 for indian_pine
    def __init__(self, in_channels=220, out_channels=220): # Note to change in_channels and out_channels based on band number
        super(RIDNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.eam1 = EAM()
        self.eam2 = EAM()
        self.eam3 = EAM()
        self.eam4 = EAM()
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        conv1 = self.conv1(x)
        eam1 = self.eam1(conv1)
        eam2 = self.eam2(eam1)
        eam3 = self.eam3(eam2)
        eam4 = self.eam4(eam3)
        conv2 = self.conv2(eam4)
        out = conv2 + x
        return out
