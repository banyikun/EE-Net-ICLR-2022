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

class NeuralUCBDiag:
    def __init__(self, dim, lamdba=1, nu=1, hidden=100):
        self.func = Network(dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,)).to(device)
        self.nu = nu

    def select(self, context):
        tensor = torch.from_numpy(context).float().to(device)
        mu = self.func(tensor)
        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        f_res = []
        ucb = []
        for fx in mu:
            #print("fx:", fx)
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            g_list.append(g)
           # print(self.lamdba)
            sigma2 = self.lamdba * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            f_res.append(fx.item())
            ucb.append(sigma.item())
            sample_r = fx.item() + sigma.item()
            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r
        arm = np.argmax(sampled)
        self.U += g_list[arm] * g_list[arm]
        return arm, f_res, ucb
    
    def update(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)

    def train(self):
        optimizer = optim.SGD(self.func.parameters(), lr=1e-2)
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
                delta = self.func(c.to(device)) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length





if torch.cuda.is_available():  
    dev = "cuda:1" 
else:  
    dev = "cpu" 
device = torch.device(dev)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NeuralUCB')
    parser.add_argument('--dataset', default='mnist', type=str, help='yelp, disin, mnist, movielens')
    args = parser.parse_args()
    
    arg_size = 1
    arg_shuffle = 1
    arg_seed = 0
    arg_nu = 1
    arg_lambda = 0.0001
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
        l = NeuralUCBDiag(b.dim, arg_lambda, arg_nu, arg_hidden)
        for t in range(10000):
            context, rwd = b.step()
            arm_select, f_res, ucb = l.select(context)
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

        print("round:", t, summ)

    
    
    
    