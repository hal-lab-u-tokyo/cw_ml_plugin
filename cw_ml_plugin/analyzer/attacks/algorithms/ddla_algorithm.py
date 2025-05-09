import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from itertools import product

from tqdm import tqdm

from chipwhisperer.common.api.ProjectFormat import Project

from cw_ml_plugin.analyzer.attacks.algorithms.ddla_models import *
from cw_ml_plugin.analyzer.attacks.algorithmsbase import AlgorithmsBase

def deep_learning(device, 
                bnum, 
                key, 
                traces:np.ndarray, 
                num_epoch, 
                batch_size, 
                learning_rate, 
                hypothesis_values,
                label_length):
    numtraces = traces.shape[0]
    trace_length = traces.shape[1]
    
    loss_result = np.zeros([num_epoch])
    accuracy_result = np.zeros([num_epoch])
    sensitivity_result = np.zeros((trace_length))
    
    #model for ASCAD
    model = MLPexp_20(trace_length, label_length).to(device=device)
    
    #model for Sakura
    # model = MLPexp_200(trace_length, label_length).to(device=device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.MSELoss().to(device=device)
    # criterion = nn.CrossEntropyLoss().to(device=device)

    for epoch in range(num_epoch):
        #For accuracy
        correct = 0
        total = 0
        
        for start_ind in range(0, numtraces, batch_size):
            optimizer.zero_grad()
            
            end_ind = start_ind + batch_size
            
            traces_itr = torch.from_numpy(traces[start_ind:end_ind]).clone().detach().to(device=device).requires_grad_(True)
            hyp_itr = torch.from_numpy(hypothesis_values[start_ind:end_ind]).clone().detach().to(device=device)
            
            input = traces_itr
            target = hyp_itr.to(torch.float32)
            input.retain_grad()
            output = model(input)
            loss = criterion(output, target)
            
            loss.backward()
            
            _, guess_class_ind = torch.max(output, 1)
            _, answer_class_ind = torch.max(target, 1)
            correct += (guess_class_ind == answer_class_ind).sum().item()
            total += target.size(0)
            
            # batch_sensitivity = (traces_itr.grad * traces_itr).sum(dim=0)
            batch_sensitivity = (traces_itr.grad * traces_itr).abs().sum(dim=0) # type: ignore
            sensitivity_result += batch_sensitivity.detach().cpu().numpy()
            
            optimizer.step()
            
        with torch.no_grad():
            loss_result[epoch] = loss.detach().cpu().numpy()
            accuracy_result[epoch] = correct / total
            
    sensitivity_result = abs(sensitivity_result)
    
    return bnum, key, loss_result, accuracy_result, sensitivity_result

def run_deep_learning(args):
    return deep_learning(*args)

class DDLA_Algorithm(AlgorithmsBase):
    _name = "DDLA_Algorithm"
    
    def __init__(self):
        
        AlgorithmsBase.__init__(self)
        
        #Need to Check if neccecary or not
        #Copying the original cw code just for now
        self.getParams().addChildren([
            {'name':'Iteration Mode', 'key':'itmode', 'type':'list', 'values':{'Depth-First':'df', 'Breadth-First':'bf'}, 'value':'bf', 'action':self.updateScript},
            {'name':'Skip when PGE=0', 'key':'checkpge', 'type':'bool', 'value':False, 'action':self.updateScript},
        ])
        self.updateScript()
        
    def addTraces(self, project:Project, parameters, label_func):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        numtraces = project._traceManager.num_traces()
        trace_length = project._traceManager.num_points()
        
        traces = np.zeros([numtraces, trace_length], dtype=np.float32)
        plaintexts = np.zeros([numtraces, 16], dtype=np.uint8)
        ciphertexts = np.zeros([numtraces, 16], dtype=np.uint8)
        knownkeys = np.zeros([numtraces, 16], dtype=np.uint8)
        
        for tnum in range(numtraces):
            traces[tnum] = project._traceManager.get_trace(tnum)
            plaintexts[tnum] = project._traceManager.get_textin(tnum)
            ciphertexts[tnum] = project._traceManager.get_textout(tnum)
            knownkeys[tnum] = project._traceManager.get_known_key(tnum)
        
        #just to get the label length
        #this could be modified
        key = bnum = 0
        pt = plaintexts[0]
        ct = ciphertexts[0]
        nk = knownkeys[0]
        state = [{'knownkey': None}] * 16
        state[bnum]['knownkey'] = nk
        label_length =  len(label_func(self.model, pt, ct, key, state, bnum))

        hypothesis_values = np.zeros([16, 256, numtraces, label_length], dtype=np.uint8)
        state = [{'knownkey': None}] * 16
        
        byte_key_combinations = list(product(parameters._byte_range, parameters._key_range))
        
        print('Calculating Hypothesis Values')
        pbar = tqdm(byte_key_combinations)
        for combination in pbar:
            bnum = combination[0]
            key = combination[1]
            for tnum in range(numtraces):
                pt = plaintexts[tnum]
                ct = ciphertexts[tnum]
                nk = knownkeys[tnum] if len(knownkeys) > 0 else None
                state[bnum]['knownkey'] = nk
                
                hypothesis_values[bnum,key,tnum,:] = label_func(self.model, pt, ct, key, state, bnum)
        
        print(f'Trainig the model with {numtraces} traces on devive {device}')
        print('CAUTION: the estimated time on the progress bar might not be accurate')
        mpj = parameters._max_parallel_jobs

        # プログレスバーの設定
        task_desc = f"Attacking byte_range {parameters._byte_range}, key_range {parameters._key_range}"

        # タスクリストの生成
        tasks = [
            (device,
            combination[0],
            combination[1],
            traces,
            parameters._epoch,
            parameters._batch_size,
            parameters._learning_rate,
            hypothesis_values[combination[0], combination[1], :, :],
            label_length)
            for combination in byte_key_combinations
        ]

        with mp.Pool(processes=mpj) as pool:
            results = list(
                tqdm(
                    pool.imap(run_deep_learning, tasks),
                    total=len(tasks),
                    desc=task_desc
                )
            )

        loss_result = np.zeros([len(parameters._byte_range), 256, parameters._epoch])
        accuracy_result = np.zeros([len(parameters._byte_range), 256, parameters._epoch])
        sensitivity_result = np.zeros([len(parameters._byte_range), 256, trace_length])
        
        for bnum, key, loss, accuracy, sensitivity in results:
            index_for_list = bnum - parameters._byte_range[0]
            loss_result[index_for_list,key,:] = loss
            accuracy_result[index_for_list,key,:] = accuracy
            sensitivity_result[index_for_list,key,:] = sensitivity
        
        
        for bnum in parameters._byte_range:
            index_for_list = bnum - parameters._byte_range[0]

            if parameters._sort_param == 'loss':
                data = loss_result[index_for_list,:,-1]
            elif parameters._sort_param == 'accuracy':
                data = accuracy_result[index_for_list,:,-1]
            elif parameters._sort_param == 'sensitivity':
                data = np.max(sensitivity_result[index_for_list,:,:], axis=1)
            else:
                raise TypeError(f"Unknown sort parameter: {parameters._sort_param}")
                
            self.stats.update_subkey(bnum, data, numtraces)  # type: ignore
            self.stats.update_loss(bnum, loss_result[index_for_list]) # type: ignore
            self.stats.update_accuracy(bnum, accuracy_result[index_for_list]) # type: ignore
            self.stats.update_metrics(bnum, sensitivity_result[index_for_list]) # type: ignore
