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

class Neural_noexplore:
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
            sampled.append(fx.item())
        arm = np.argmax(sampled)
        return arm
    
    def update(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)

    def train(self, t):
        optimizer = optim.Adam(self.func.parameters(), lr=1e-4)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural NoExploration')
    parser.add_argument('--dataset', default='mnist', type=str, help='mnist, cifar10, notmnist, yelp')
    args = parser.parse_args()
    regrets_all = []
    arg_nu = 1
    arg_lambda = 0.0001
    for i in range(10):
        if args.dataset == "yelp":
            b = load_yelp()
            arg_nu = 1 
            arg_lambda = 0.0001
        elif args.dataset == "mnist":
            b = load_mnist_1d()
            arg_nu = 1
            arg_lambda = 0.0001

        elif args.dataset == "disin":
            b = load_disin()
        elif args.dataset == "movielens":
            b = load_movielen()
        else:
            b = Bandit_multi(dataset, 1, 0)

        summ = 0
        regrets = []
        l = Neural_noexplore(b.dim, arg_lambda, arg_nu, 100)
        for t in range(10000):
            context, rwd = b.step()
            arm_select = l.select(context)
            r = rwd[arm_select]
            reg = np.max(rwd) - r
            summ+=reg
            l.update(context[arm_select], r)
            if t<1000:
                if t%10 == 0:
                    loss = l.train(t)
            else:
                if t%100 == 0:
                    loss = l.train(t)
            regrets.append(summ)
            if t % 50 == 0:
                print('{}: {:}, {:.4f}, {:.4f}'.format(t, summ, summ/(t+1), loss))
        print("round:", t, summ)
        regrets_all.append(regrets)



