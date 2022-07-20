from packages import *


class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

class Neural_epsilon:
    def __init__(self, dim, p = 0.01, hidden=100):
        self.func = Network(dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.p = p  # probability of making exploration
        self.lr = 0.01

    def select(self, context):
        tensor = torch.from_numpy(context).float().to(device)
        mu = self.func(tensor)
        res = []
        for fx in mu:
            res.append(fx.item())
        epsilon = np.random.binomial(1, self.p)
        if epsilon:
            arm = np.random.choice(len(context), 1)[0]
        else:
            arm = np.argmax(res)
        return arm
    
    def update(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)

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
                c = self.context_list[idx].to(device)
                r = self.reward[idx]
                delta = self.func(c) - r
                loss = delta * delta
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 2000:
                    return tot_loss / 2000
            if batch_loss / length <= 1e-3:
                return batch_loss / length