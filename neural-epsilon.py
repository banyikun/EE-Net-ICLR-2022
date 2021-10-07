from packages import *
from load_data import load_yelp, Bandit_multi, load_mnist_1d, load_movielen, load_disin, load_disin_20


class fcn(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(fcn, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

class Neural_epsilon:
    def __init__(self, dim, p = 0.1, lamdba=1, nu=1, hidden=100):
        self.func = fcn(dim, hidden_size=hidden)
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,))
        self.nu = nu
        self.p = p  # probability of making exploration

    def select(self, context):
        tensor = torch.from_numpy(context).float()
        mu = self.func(tensor)
        res = []
        for fx in mu:
            res.append(fx.item())
        epsilon = np.random.binomial(1, self.p)
        if epsilon:
            #print("random")
            arm = np.random.choice(len(context), 1)
        else:
            #print("greedy")
            arm = np.argmax(res)
        return arm, res
    
    def update(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)

    def train(self):
        optimizer = optim.SGD(self.func.parameters(), lr=1e-3)
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
                delta = self.func(c) - r
                loss = delta * delta
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 10000:
                    return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length
            
            
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural-epsilon')
    parser.add_argument('--dataset', default='mnist', type=str, help='yelp, disin, mnist, movielens')
    args = parser.parse_args()
    


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
        l = Neural_epsilon(dim = b.dim)
        for t in range(10000):
            context, rwd = b.step()
            arm_select, res = l.select(context)
            r = rwd[arm_select]
            if not np.isscalar(r):
                r= r[0]
            reg = np.max(rwd) - r
            summ+=reg
            l.update(context[arm_select], r)
            if t<2000:
                if t%10 == 0:
                    #print(rwd)
                    loss = l.train()
            else:
                if t%100 == 0:
                    loss = l.train()
            regrets.append(summ)
            if t % 50 == 0:
                print('{}: {:}, {:.3}, {:.3e}'.format(t, summ, summ/(t+1), loss))

        print("round:", t, summ)

    
    
    
    