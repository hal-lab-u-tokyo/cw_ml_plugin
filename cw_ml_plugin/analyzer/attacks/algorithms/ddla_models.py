import torch
import torch.nn as nn
import math

label_length = 2

class MLPsim(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(50, 70),  #* end_ind - start_ind
            nn.ReLU(inplace=True),
            nn.Linear(70, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50,2),
            nn.Softmax()
        )
    def forward(self, x):
        output = self.network(x)
        return output

class CNNsim(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 8, 8), #* 入力は50*1、出力は43*8
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2), #* 入力は43*8,出力は22*4
            nn.Conv1d(4, 4, 1), #* 入力は22*4,出力は19*4
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2), #* 入力は19*4,出力は10*2
            nn.Flatten(),
            nn.Linear(10*2, 2),
            nn.Softmax()
        )
    def forward(self, x):
        output = self.network(x)
        return output

class MLPexp_20(nn.Module):
    def __init__(self, num_time, label_length):
        super().__init__()
        self.num_time = num_time
        self.label_length = label_length
        self.network = nn.Sequential(
            nn.Linear(self.num_time, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, self.label_length),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        output = self.network(x)
        return output

class MLPexp_200(nn.Module):
    def __init__(self, num_time, label_length):
        super().__init__()
        self.num_time = num_time
        self.label_length = label_length
        self.network = nn.Sequential(
            nn.Linear(self.num_time, 200),
            nn.SELU(),
            nn.Linear(200, 80),
            nn.SELU(),
            nn.Linear(80, 16),
            nn.SELU(),
            nn.Linear(16, self.label_length),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        output = self.network(x)
        return output
    
# class MLPexp_200(nn.Module):
#     def __init__(self, num_time, label_length):
#         super().__init__()
#         self.num_time = num_time
#         self.label_length = label_length
#         self.network = nn.Sequential(
#             nn.Linear(self.num_time, 200),
#             nn.SELU(),
#             nn.Linear(200, 200),
#             nn.SELU(),
#             nn.Linear(200, 200),
#             nn.SELU(),
#             nn.Linear(200, 80),
#             nn.SELU(),
#             nn.Linear(80, 16),
#             nn.SELU(),
#             nn.Linear(16, self.label_length),
#             nn.Softmax(dim=1)
#         )
#     def forward(self, x):
#         output = self.network(x)
#         return output
    
class CNNexp(nn.Module):
    def __init__(self, num_time, label_length):
        super().__init__()
        self.num_time = num_time
        self.label_length = label_length
        self.network = nn.Sequential(
            nn.Conv1d(1, 4, 32), #* 入力は1*end-start、出力は4*end-start-31
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=2), #* 入力は4*end-start-31,出力は4*math.ceil((end-start-31)/2)
            nn.BatchNorm1d(4),
            nn.Conv1d(4, 4, 16), #* 入力は4*(math.ceil((end-start-31)/2)),出力は4*(math.ceil((end-start-31)/2)-15)
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4), #* 入力は4*(4*math.ceil((end-start-31)/2)-15),出力は4*math.ceil((math.ceil((end-start-31)/2)-15)/4)
            nn.BatchNorm1d(4),
            nn.Flatten(),
        )

        self.network2 = nn.Sequential(
            nn.Linear(4*math.floor((math.floor((self.num_time - 31)/2)-15)/4), self.label_length),
            nn.Softmax()
        )
    def forward(self, x):
        output = self.network(x)
        output = output.view(output.size(0),-1)
        output = self.network2(output)
        return output


class MLPmod(nn.Module):
    def __init__(self, pretrained, num_points, device):
        super().__init__()
        self.device = device
        self.pretrained = pretrained
        pretrained = pretrained.to(device=self.device).requires_grad_(False)
        self.base = nn.Sequential(
            nn.Linear(num_points, 200),
            nn.SELU(),
            nn.Linear(200, 200),
            nn.SELU(),
            nn.Linear(200, 200),
            nn.SELU(),
            nn.Linear(200, 80),
            nn.SELU(),
            nn.Linear(80, 16),
            nn.SELU(),
            nn.Linear(16, label_length),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        output = torch.tensor([]).to(device=self.device)
        output = self.pretrained(x)
        output = self.base(output)
        return output

class MLPpara(nn.Module):
    def __init__(self, pretrained, num_points, device):
        super().__init__()
        self.pretrained = pretrained
        self.device = device
        for i in range(256):
            self.pretrained[i] = self.pretrained[i].requires_grad_(False)
        self.share = nn.Sequential(
            nn.Linear(num_points, 200),
            nn.SELU(),
        )
        self.base = nn.Sequential(
            nn.Linear(200, 20),
            nn.SELU(),
            nn.Linear(20, label_length),
            nn.Softmax(dim=1)
        )
        self.base_set = [self.base for i in range(256)]

    def forward(self, x, magnification):
        output = torch.tensor([]).to(device=self.device)
        temp = torch.tensor([]).to(device=self.device)
        for i in range(256):
            mag = magnification[i, :]
            mag = mag.view(-1, 1)
            temp = mag * x
            temp = self.pretrained[i](temp)
            temp = self.share(temp)
            temp = self.base_set[i](temp)
            output = torch.cat([output, temp], dim=0)
        return output