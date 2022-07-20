from packages import *


class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

## flatten a large tuple containing tensors of different sizes
def flatten(tensor):
    T=torch.tensor([]).to(device)
    for element in tensor:
        T=torch.cat([T,element.to(device).flatten()])
    return T
    
#concatenation of all the parameters of a NN
def get_theta(model):
    return flatten(model.parameters())


class NeuralTS:
    """Neural Thompson Sampling Strategy"""
    def __init__(self,dim, n_arm, m,reg=1,sigma=1,nu=0.15):
        self.K = n_arm 
        self.nu=nu
        self.sigma=sigma
        self.m = m
        self.d = dim
        self.estimator=Network(self.d, hidden_size=m).to(device)
        self.optimizer = torch.optim.SGD(self.estimator.parameters(), lr = 0.01)
        self.current_loss=0
        self.t=1
        self.total_param = sum(p.numel() for p in self.estimator.parameters() if p.requires_grad)
        self.Design =  torch.ones((self.total_param,)).to(device)
        self.rewards=[]
        self.context_list = []
        
        
    def select(self, context):
        self.features=torch.from_numpy(context).float().to(device)
        estimated_rewards=[]
        
        sigma_l = []
        f_l = []
        for k in range(self.K):
            f=self.estimator(self.features[k])
            g=torch.autograd.grad(outputs=f,inputs=self.estimator.parameters())
            g=flatten(g).detach()            
            sigma2 = g * g /self.Design
            sigma = torch.sqrt(torch.sum(sigma2)) * self.nu
            sample = torch.normal(mean=f.item(), std=sigma)
            estimated_rewards.append(sample.item())
            sigma_l.append(sigma)
            f_l.append(f.item())
        
        arm_to_pull=np.argmax(estimated_rewards)
        
        return arm_to_pull 
    
    def update(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        new_context = torch.from_numpy(context.reshape(1, -1)).float().to(device)
        self.rewards.append(reward)
        f_t=self.estimator(new_context)
    
        g=torch.autograd.grad(outputs=f_t,inputs=self.estimator.parameters())
        
        g=flatten(g)
        g=g/(np.sqrt(self.m))
        self.Design+=torch.matmul(g,g.T).to(device)
        self.t+=1


    def train(self, t):
        length = len(self.rewards)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx].to(device)
                r = self.rewards[idx]
                delta = self.estimator(c) - r
                self.current_loss = delta * delta
                self.optimizer.zero_grad() 
                    #gradient descent
                if self.t==1:
                    self.current_loss.backward(retain_graph=True)    
                else:
                    self.current_loss.backward()

                self.optimizer.step() 
                batch_loss +=  self.current_loss.item()
                tot_loss +=  self.current_loss.item()
                cnt += 1
                if cnt >= 2000:
                    return tot_loss / 2000
            if batch_loss / length <= 1e-3:
                return batch_loss / length
      
    
                