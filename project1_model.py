# from numpy import short
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary


N = 4
B_i = [3,4,4,3]         # Number of blocks in each layer
C_i = [32,64,128,256]       # Number of channels in Residual Layer i
F_i = [3, 3, 3, 3]            # Conv. kernel size in Residual Layer i
K_i = [1, 1, 1, 1]            # Shortcut kernel size
P = 3                       # Average pool kernel size



class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, shortcut_kernel_size, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=shortcut_kernel_size, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, C_i, B_i, F_i, K_i, num_classes=10):
        super(ResNet, self).__init__()
        # First layer
        self.inchannel = C_i[0]#C1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, C_i[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_i[0]), #bn(#channel)
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, C_i[0], B_i[0], F_i[0], K_i[0], stride=1)
        self.layer2 = self.make_layer(ResidualBlock, C_i[1], B_i[1], F_i[1], K_i[1], stride=2)
        self.layer3 = self.make_layer(ResidualBlock, C_i[2], B_i[2], F_i[2], K_i[2], stride=2)
        self.layer4 = self.make_layer(ResidualBlock, C_i[3], B_i[3], F_i[3], K_i[3], stride=2)

        self.fc = nn.Linear(C_i[3], num_classes)

    def make_layer(self, block, channels, num_blocks, kernel_size, shortcut_kernel_size, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[stride,1,...]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, kernel_size, shortcut_kernel_size, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, P)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# L1 norm
# class Regularization(torch.nn.Module):
#     def __init__(self,model,weight_decay,p=2):
#         '''
#         :param model 
#         :param weight_decay
#         :param p: exponent value (p = 0 -> L2 norm, p = 1 -> L1 norm)
#         '''
#         super(Regularization, self).__init__()
#         if weight_decay <= 0:
#             #print("param weight_decay can not <=0")
#             exit(0)
#         self.model=model
#         self.weight_decay=weight_decay
#         self.p=p
#         self.weight_list=self.get_weight(model)
#         #self.weight_info(self.weight_list)

#     def to(self,device):
#         '''
#         :param device: cude or cpu
#         :return:
#         '''
#         self.device=device
#         super().to(device)
#         return self

#     def forward(self, model):
#         self.weight_list=self.get_weight(model)
#         reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
#         return reg_loss

#     def get_weight(self,model):
#         '''
#         :param model:
#         :return:
#         '''
#         weight_list = []
#         for name, param in model.named_parameters():
#             if 'weight' in name:
#                 weight = (name, param)
#                 weight_list.append(weight)
#         return weight_list

#     def regularization_loss(self,weight_list, weight_decay, p=2):
#         '''
#         :param weight_list:
#         :param p:
#         :param weight_decay:
#         :return:
#         '''
#         # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
#         # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
#         # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
#         # reg_loss=torch.FloatTensor([0.]).to(self.device)
#         reg_loss=0
#         for name, w in weight_list:
#             l2_reg = torch.norm(w, p=p)
#             reg_loss = reg_loss + l2_reg

#         reg_loss=weight_decay*reg_loss
#         return reg_loss


def project1_model():
    return ResNet(ResidualBlock, C_i, B_i, F_i, K_i)

# # calculate the parameters
# if __name__ == "__main__":
    
#     net = project1_model()
#     net.cuda()
#     summary(net,(3,32,32))
