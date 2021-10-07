from packages import *
from load_data import load_yelp, Bandit_multi, load_mnist_1d, load_movielen, load_disin, load_disin_20


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
        self.optimizer = torch.optim.Adam(self.estimator.parameters(), lr = 10**(-4))
        self.current_loss=0
        self.t=1
        self.total_param = sum(p.numel() for p in self.estimator.parameters() if p.requires_grad)
        self.Design =  torch.ones((self.total_param,)).to(device)
        self.rewards=[]
        self.context_list = []
        
        
    def select(self, context):
        self.features=torch.from_numpy(context).float().to(device)
        estimated_rewards=[]
        
        for k in range(self.K):
            f=self.estimator(self.features[k])
            #print("f:", f.item())
            g=torch.autograd.grad(outputs=f,inputs=self.estimator.parameters())
            g=flatten(g).detach()            
            #sigma_squared=(torch.matmul(torch.matmul(g.T,self.DesignInv),g)).to(device)
            sigma2 = g * g /self.Design
            sigma = torch.sqrt(torch.sum(sigma2)) * self.nu
            
            #print("sigma:", sigma.item())
            #r_tilda=(self.nu)*(sigma)*torch.randn(1).to(device)+f.detach()
            sample = torch.normal(mean=f.item(), std=sigma)
            #print("sample", sample.item())
            estimated_rewards.append(sample.item())
        
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


    def train(self):
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
                if self.t==1:
                    self.current_loss.backward(retain_graph=True)    
                else:
                    self.current_loss.backward()

                self.optimizer.step() 
                batch_loss +=  self.current_loss.item()
                tot_loss +=  self.current_loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length

            
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NeuralTS')
    parser.add_argument('--dataset', default='mnist', type=str, help='yelp, disin, mnist, movielens')
    args = parser.parse_args()
    
    arg_size = 1
    arg_shuffle = 1
    arg_seed = 0
    arg_nu = 1
    arg_lambda = 0.01
    arg_hidden = 100

    for i in range(10):
        if args.dataset == "yelp":
            b = load_yelp()
        elif args.dataset == "disin":
            b = load_disin()
        elif args.dataset == "mnist":
            b = load_mnist_1d()
        elif args.dataset == "movielens":
            b = load_movielen()
        else:
            b = Bandit_multi(dataset, 1, 0)
        
    summ = 0
    regrets = []
    l = NeuralTS(b.dim, b.n_arm, 100)
    for t in range(10000):
        context, rwd = b.step()
        arm_select = l.select(context)
        r = rwd[arm_select]
        reg = np.max(rwd) - r
        summ+=reg
        l.update(context[arm_select], r)
        if t<2000:
            if t%10 == 0:
                loss = l.train()
        else:
            if t%100 == 0:
                loss = l.train()
        regrets.append(summ)
        if t % 50 == 0:
            print('{}: {:}, {:.3}, {:.3e}'.format(t, summ, summ/(t+1), loss))
            
    
    
    
    
                