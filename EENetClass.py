
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from skimage.measure import block_reduce




if torch.cuda.is_available():  
    dev = "cuda:1" 
else:  
    dev = "cpu" 
device = torch.device(dev)


'''Network Structure'''

class Network_exploitation(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_exploitation, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))
    
    
class Network_exploration(nn.Module):
    def __init__(self, input_dim, _kernel_size = 100,  _stride = 50,  _channels = 1):
        super(Network_exploration, self).__init__()
        self.conv1 = nn.Conv1d(1, _channels, kernel_size = _kernel_size, stride = _stride)
        _num_dim = int(((input_dim - _kernel_size)/ _stride + 1) * _channels)
        self.fc1 = nn.Linear(_num_dim, 100)
        self.fc2 = nn.Linear(100, 1)
        self.activate = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.activate(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Network_decision_maker(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_decision_maker, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))
    
    

'''Network functions'''

class Exploitation:
    def __init__(self, input_dim, num_arm, pool_step_size, lr = 0.01, hidden=100):
        '''input_dim: number of dimensions of input'''    
        '''num_arm: number of arms'''
        '''lr: learning rate'''
        '''hidden: number of hidden nodes'''
        
        self.func = Network_exploitation(input_dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []        
        self.lr = lr
        self.pool_step_size = pool_step_size
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
    
    
    def update(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)
        
        
    def output_and_gradient(self, context):
        tensor = torch.from_numpy(context).float().to(device)
        results = self.func(tensor)
        g_list = []
        res_list = []
        for fx in results:
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            g_list.append(np.array(g.cpu()))
            res_list.append([fx.item()])
        res_list = np.array(res_list)
        g_list = np.array(g_list)
        
        #'''Gradient Aggregation'''
        g_list = block_reduce(g_list, block_size=(1, self.pool_step_size), func=np.mean)
        return res_list, g_list
    
    def return_gradient(self, context):
        tensor = torch.from_numpy(context).float().to(device)
        result = self.func(tensor)
        self.func.zero_grad()
        result.backward(retain_graph=True)
        g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
        g = [np.array(g.cpu())]
        g = np.array(g)
        g_aggre = block_reduce(g, block_size=(1, self.pool_step_size), func=np.mean)
        return result.item(), g_aggre[0]
    
    def train(self):
        optimizer = optim.SGD(self.func.parameters(), lr=self.lr)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                optimizer.zero_grad()
                loss = (self.func(c.to(device)) - r)**2
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 2000:
                    return tot_loss / cnt
            if batch_loss / length <= 1e-3:
                return batch_loss / length

            

    
class Exploration:
    def __init__(self, input_dim, lr=0.001):
        self.func = Network_exploration(input_dim=input_dim).to(device)
        self.context_list = []
        self.reward = []
        self.lr = lr

    
    def update(self, context, reward):
        tensor = torch.from_numpy(context).float().to(device)
        tensor = torch.unsqueeze(tensor, 0)
        tensor = torch.unsqueeze(tensor, 0)
        self.context_list.append(tensor)
        self.reward.append(reward)
        
    def output(self, context):
        tensor = torch.from_numpy(context).float().to(device)
        tensor = torch.unsqueeze(tensor, 1)
        
        ress = self.func(tensor).cpu()
        res = ress.detach().numpy()
        return res
    

    def train(self):
        optimizer = optim.Adam(self.func.parameters(), lr=self.lr, weight_decay=0.0001)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                output = self.func(c.to(device))
                optimizer.zero_grad()
                delta = self.func(c.to(device)) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 2000:
                    return tot_loss / cnt
            if batch_loss / length <= 2e-3:
                #print("batched loss",  batch_loss / length)
                return batch_loss / length     
            
            


class Decision_maker:
    def __init__(self, input_dim, hidden=20, lr = 0.01):
        self.func = Network_decision_maker(input_dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.loss = nn.BCEWithLogitsLoss()
        self.lr = lr
        print("f3_lr", self.lr)

    
    def update(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)
        
        
    def select(self, context):
        tensor = torch.from_numpy(context).float().to(device)
        ress = self.func(tensor).cpu()
        res = ress.detach().numpy()
        return np.argmax(res)

    def train(self):
        optimizer = optim.Adam(self.func.parameters(), lr=self.lr)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                target = torch.tensor([r]).unsqueeze(1).to(device)
                output = self.func(c.to(device))
                loss = (output - r)**2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / cnt
            if batch_loss / length <= 1e-3:
                return batch_loss / length                   
    
