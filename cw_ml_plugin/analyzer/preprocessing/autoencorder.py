import numpy as np
import torch.nn as nn
import torch.optim as optim
import math
import torch
import chipwhisperer as cw
from typing import Optional, Union
from chipwhisperer.common.api.ProjectFormat import Project

learning_rate = 0.001
class Auto_Encorder():
    def __init__(self, Input_Project: Project, Target_Project: Project, hook_layer_num = None):
        """
            Runs the AutoEncorder
            Don't forget to set the parameters below.
            
            parameters
            Input Project : the input trace data
            Output Project : the label trace data
            epoch: the number of times the entire dataset is passed through the model during training
            batch size: the number of training examples utilized in one iteration
            hool_layer_num: the number of the layer you want to hook. 0 means the first layer.
        """
        #Initialize the GPU
        #required for Intel MacOS
        torch.set_num_threads(1) # to avoid "UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown warnings.warn('resource_tracker: There appear to be %d '"
        torch.cuda.empty_cache()
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self._input_project: Project = Input_Project
        self._target_project: Project = Target_Project
        self._epoch:Optional[int] = None
        self._batch_size:int = 500
        self._name:str = ""
        self._input = None
        self._target = None
        self._output = None
                
        if not len(Input_Project.traces) > 0:
            raise ValueError(f'traces are empty')
        elif not len(Input_Project.traces) == len(Target_Project.traces):
            raise ValueError(f'trace number of input and target project is not same')
        else:
            self._trace_num= len(Input_Project.traces)
            self._perm = np.random.permutation(self._trace_num)

            inverse_perm = np.empty_like(self._perm)
            inverse_perm[self._perm] = np.arange(self._trace_num)
            self._inv_perm = inverse_perm
            
        if not len(Input_Project.waves[0]) > 0:
            raise ValueError(f'trace data is empty')
        else: 
            self._trace_length = len(self._input_project.waves[0])
            
        self.hook_layer_num = hook_layer_num
        self.activations = {}
            
            
        
    def _set_epoch(self, epoch:int) -> None:
        self._epoch = epoch
        
    @property
    def epoch(self) -> Optional[int]:
        return self._epoch
    
    @epoch.setter
    def epoch(self, epoch:int):
        """Set the epoch for autoencording.
            Default is set at 500.
        """
        if not isinstance(epoch, int):
            raise TypeError(f'expected int; got{type(epoch)}')
        self._set_epoch(epoch)
        
    def _set_batch_size(self, batch_size:int) -> None:
        self._batch_size = batch_size
    
    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, batch:int) -> None:
        """Set the batch size for autoencording"""
        if not isinstance(batch, int):
            raise TypeError(f'expected int; got {type(batch)}')
        self._set_batch_size(batch)
    
    def _set_name(self, name:str) -> None:
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, name) -> None:
        """If you want to save the output project set the path.
            If not imput a random name
        """
        if not isinstance(name, str):
            raise TypeError(f'expected str; got {type(name)}')
        self._set_name(name)
    
    @property
    def trace_length(self) -> Optional[int]:
        return self._trace_length

    @property
    def trace_num(self) -> Optional[int]:
        return self._trace_num
    
    def hook_fn(self, module, input, output):
        self.activations[module] = output
    
    def _make_trace_matrix(self, proj: Project) -> np.ndarray:
        if proj.waves[0] is None:
            raise ValueError('Trace Data Unfound')
        self._trace_length = len(proj.waves[0])
        trace_matrix = np.zeros((self._trace_num, self._trace_length), dtype=np.float32)
        for trace_num in range(self._trace_num):
            trace = np.array(proj.waves[trace_num], dtype=np.float32)
            if trace.shape[0] != self._trace_length:
                raise ValueError(f'Trace length mismatch at index {trace_num}')
            trace_matrix[trace_num] = trace
        return trace_matrix

    def autoencord(self):
        model = MLPpropose(self._trace_length).to(device=self._device)
        criterion = nn.MSELoss().to(device=self._device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
        torch.manual_seed(0)

        self._input = self._make_trace_matrix(self._input_project)[self._perm, :]
        self._target = self._make_trace_matrix(self._target_project)[self._perm, :]
        input = torch.from_numpy(self._input.astype(np.float32)).to(self._device)
        target = torch.from_numpy(self._target.astype(np.float32)).to(self._device)

        assert input.shape == target.shape
        assert input.dtype == torch.float32

        batch_itr = self._trace_num // self._batch_size
        for epoch_itr in range(self._epoch):
            for itr in range(batch_itr):
                batch_input = input[itr*self._batch_size:(itr+1)*self._batch_size, :].clone().detach()
                batch_input = batch_input.to(self._device)
                batch_input.requires_grad = True

                batch_target = target[itr*self._batch_size:(itr+1)*self._batch_size, :].clone().detach().to(self._device)

                batch_output = model(batch_input)
                loss = criterion(batch_output, batch_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'epoch:{epoch_itr} Loss:{loss}')
        return model
    
    def run(self):
        model = self.autoencord()
        input = torch.from_numpy(self._input).to(device = self._device, dtype = torch.float32)
        output_trace = model(input).detach().cpu().numpy()
        
        if self._name == "":
            proj = Project()
            # print(f'Cannot Save The AutoEncord Result, Data will be unalbe to reach post analysis')
        else:
            proj = cw.create_project(self._name)
            
        for num in range(self._trace_num):
            value = output_trace[self._inv_perm[num]]
            plain = self._input_project.textins[num]
            enc = self._input_project.textouts[num]
            key = self._input_project.keys[num]
            proj.traces.append(cw.Trace(value, plain, enc, key)) # type: ignore
        
        return proj
    
    def run_interpert(self):
        if self.hook_layer_num is None:
            raise ValueError(f'hook_layer_num is not set')
        else:
            model = self.autoencord()
            
            #Code using hook function
            # if self.hook_layer_num is not None:
            #     specific_layer = model.network[self.hook_layer_num]
            #     specific_layer.register_forward_hook(self.hook_fn)

            # input = torch.from_numpy(self._input).to(device = self._device, dtype = torch.float32)
            # output_trace = model(input).detach().cpu().numpy()
            # print(f"Specific Layer Output: {self.activations[specific_layer]}")
            
            #Code without the hook function
            partial_model = torch.nn.Sequential(*list(model.network.children())[:self.hook_layer_num + 1])

            input = torch.from_numpy(self._input).to(device=self._device, dtype=torch.float32)

            intermediate_output = partial_model(input).detach().cpu().numpy()

            # print(f"Intermediate Output Shape: {intermediate_output.shape}")
            if self._name == "":
                proj = Project()
                # print(f'Cannot Save The AutoEncord Result, Data will be unalbe to reach post analysis')
            else:
                proj = cw.create_project(self._name)
                
            for num in range(self._trace_num):
                value = intermediate_output[self._inv_perm[num]]
                plain = self._input_project.textins[num]
                enc = self._input_project.textouts[num]
                key = self._input_project.keys[num]
                proj.traces.append(cw.Trace(value, plain, enc, key)) # type: ignore
            return proj
        
class MLPpropose(nn.Module):
    def __init__(self, trace_length):
        super().__init__()
        self.trace_length = trace_length
        self.network = nn.Sequential(
            nn.Linear(self.trace_length, math.floor((self.trace_length) * 3 / 4)),
            nn.SELU(),
            nn.Linear(math.floor((self.trace_length) * 3 / 4), math.floor((self.trace_length)/2)),
            nn.SELU(),
            nn.Linear(math.floor((self.trace_length)/2), math.floor((self.trace_length)* 3 / 4)),
            nn.SELU(),
            nn.Linear(math.floor((self.trace_length) * 3 / 4), self.trace_length),
        )
        
    def forward(self, x):
        output = self.network(x)
        return output
    

class MLP_20per_5(nn.Module):
    def __init__(self, trace_length):
        super().__init__()
        self.trace_length = trace_length
        
        # 各層のノード数を計算
        self.layer_1 = trace_length
        self.layer_2 = math.floor(trace_length * 0.6)
        self.layer_3 = math.floor(trace_length * 0.2)
        self.layer_4 = math.floor(trace_length * 0.6)
        self.layer_5 = trace_length
        
        self.network = nn.Sequential(
            nn.Linear(self.layer_1, self.layer_2),  # 100% → 60%
            nn.SELU(),
            nn.Linear(self.layer_2, self.layer_3),  # 60% → 20%
            nn.SELU(),
            nn.Linear(self.layer_3, self.layer_4),  # 20% → 60%
            nn.SELU(),
            nn.Linear(self.layer_4, self.layer_5),  # 60% → 100%
        )
        
    def forward(self, x):
        output = self.network(x)
        return output

class MLP_60per_5(nn.Module):
    def __init__(self, trace_length):
        super().__init__()
        self.trace_length = trace_length
        
        # 各層のノード数を計算
        self.layer_1 = trace_length
        self.layer_2 = math.floor(trace_length * 0.8)
        self.layer_3 = math.floor(trace_length * 0.6)
        self.layer_4 = math.floor(trace_length * 0.8)
        self.layer_5 = trace_length
        
        self.network = nn.Sequential(
            nn.Linear(self.layer_1, self.layer_2),  # 100% → 80%
            nn.SELU(),
            nn.Linear(self.layer_2, self.layer_3),  # 80% → 60%
            nn.SELU(),
            nn.Linear(self.layer_3, self.layer_4),  # 60% → 80%
            nn.SELU(),
            nn.Linear(self.layer_4, self.layer_5),  # 80% → 100%
        )
        
    def forward(self, x):
        output = self.network(x)
        return output
    

class MLP_20per_9(nn.Module):
    def __init__(self, trace_length):
        super().__init__()
        self.trace_length = trace_length
        
        # 各層のノード数を計算
        self.layer_1 = trace_length
        self.layer_2 = math.floor(trace_length * 0.8)  
        self.layer_3 = math.floor(trace_length * 0.6)  
        self.layer_4 = math.floor(trace_length * 0.4)  
        self.layer_5 = math.floor(trace_length * 0.2)  
        self.layer_6 = math.floor(trace_length * 0.4)  
        self.layer_7 = math.floor(trace_length * 0.6)  
        self.layer_8 = math.floor(trace_length * 0.8)  
        self.layer_9 = trace_length  
        
        self.network = nn.Sequential(
            nn.Linear(self.layer_1, self.layer_2),  # 100% → 80%
            nn.SELU(),
            nn.Linear(self.layer_2, self.layer_3),  # 80% → 60%
            nn.SELU(),
            nn.Linear(self.layer_3, self.layer_4),  # 60% → 40%
            nn.SELU(),
            nn.Linear(self.layer_4, self.layer_5),  # 40% → 20%
            nn.SELU(),
            nn.Linear(self.layer_5, self.layer_6),  # 20% → 40%
            nn.SELU(),
            nn.Linear(self.layer_6, self.layer_7),  # 40% → 60%
            nn.SELU(),
            nn.Linear(self.layer_7, self.layer_8),  # 60% → 80%
            nn.SELU(),
            nn.Linear(self.layer_8, self.layer_9),  # 80% → 100%
        )
        
    def forward(self, x):
        output = self.network(x)
        return output

class MLP_001(nn.Module):
    def __init__(self, trace_length):
        super().__init__()
        self.trace_length = trace_length
        
        # 各層のノード数を計算
        self.layer_1 = trace_length
        self.layer_2 = math.floor(trace_length * 0.6)  
        self.layer_3 = math.floor(trace_length * 0.2)  
        self.layer_4 = math.floor(trace_length * 0.05)  
        self.layer_5 = math.floor(trace_length * 0.01)  
        self.layer_6 = math.floor(trace_length * 0.05)  
        self.layer_7 = math.floor(trace_length * 0.2)  
        self.layer_8 = math.floor(trace_length * 0.6)  
        self.layer_9 = trace_length  
        
        self.network = nn.Sequential(
            nn.Linear(self.layer_1, self.layer_2),  # 100% → 80%
            nn.SELU(),
            nn.Linear(self.layer_2, self.layer_3),  # 80% → 60%
            nn.SELU(),
            nn.Linear(self.layer_3, self.layer_4),  # 60% → 40%
            nn.SELU(),
            nn.Linear(self.layer_4, self.layer_5),  # 40% → 20%
            nn.SELU(),
            nn.Linear(self.layer_5, self.layer_6),  # 20% → 40%
            nn.SELU(),
            nn.Linear(self.layer_6, self.layer_7),  # 40% → 60%
            nn.SELU(),
            nn.Linear(self.layer_7, self.layer_8),  # 60% → 80%
            nn.SELU(),
            nn.Linear(self.layer_8, self.layer_9),  # 80% → 100%
        )
        
    def forward(self, x):
        output = self.network(x)
        return output

class Conv(nn.Module):
    def __init__(self, trace_length):
        super(Conv, self).__init__()

        self.trace_length = trace_length
        
        # Encoder (1D for time series)
        self.encoder = nn.Sequential(
            # 入力は1チャネルの時系列データ（例: (batch_size, 1, time_steps)）
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),  # 畳み込み層
            nn.BatchNorm1d(64),  # バッチ正規化
            nn.ReLU(),  # ReLU活性化関数
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),  # プーリング層

            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),  # 畳み込み層
            nn.BatchNorm1d(32),  # バッチ正規化
            nn.ReLU(),  # ReLU活性化関数
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),  # プーリング層

            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),  # 畳み込み層
            nn.BatchNorm1d(16),  # バッチ正規化
            nn.ReLU(),  # ReLU活性化関数
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),  # プーリング層
        )

        # Decoder (1D for time series)
        self.decoder = nn.Sequential(
            # デコーダー部分の処理
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # アップサンプリング

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # アップサンプリング

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # アップサンプリング

            nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1),  # 出力層
            nn.Sigmoid()  # 出力を0～1の範囲に収める
        )

    def forward(self, x):
        x = self.encoder(x)  # エンコーダーを通す
        x = self.decoder(x)  # デコーダーを通す
        return x