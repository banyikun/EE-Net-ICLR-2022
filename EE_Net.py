from packages import *


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
    def __init__(self, dim, hidden_size=100):
        super(Network_exploration, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))
    
class Network_decision_maker(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_decision_maker, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x_1 = self.activate(self.activate(self.fc1(x)))
        return self.fc2(x_1)
    
    

'''Network functions'''

class Exploitation:
    def __init__(self, dim, n_arm, lr = 0.01, hidden=100):
        '''dim: number of dimensions of input'''    
        '''n_arm: number of arms'''
        '''lr: learning rate'''
        '''hidden: number of hidden nodes'''
        
        self.func = Network_exploitation(dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        
        '''Embed gradient for exploration'''
        self.embedding = LocallyLinearEmbedding(n_components=(n_arm-1))
        
        self.lr = lr
    
    
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
        
        '''Gradient embeddings'''
        g_list = self.embedding.fit_transform(g_list)
        return res_list, g_list
    
    
    def train(self, t):
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
    def __init__(self, dim, hidden=100, lr=0.01):
        self.func = Network_exploration(dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.lr = lr
    
    def update(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)
        
    def output(self, context):
        tensor = torch.from_numpy(context).float().to(device)
        ress = self.func(tensor).cpu()
        res = ress.detach().numpy()
        return res
    

    def train(self,t):
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
                output = self.func(c.to(device))
                optimizer.zero_grad()
                delta = self.func(c.to(device)) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / cnt
            if batch_loss / length <= 1e-3:
                return batch_loss / length     
            
            


class Decision_maker:
    def __init__(self, dim, hidden=20, lr = 0.01):
        self.func = Network_decision_maker(dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.loss = nn.BCEWithLogitsLoss()
        self.lr = lr
    
    def update(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)
        
        
    def select(self, context):
        tensor = torch.from_numpy(context).float().to(device)
        ress = self.func(tensor).cpu()
        res = ress.detach().numpy()
        return np.argmax(res)

    def train(self, t):
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
    
