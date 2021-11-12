from packages import *
from load_data import load_yelp, Bandit_multi, load_mnist_1d, load_movielen, load_disin, load_disin_20



class Network_1(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_1, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

class Network_1_fun:
    def __init__(self, dim, n_arm, lr = 0.01, hidden=100):
        self.func = Network_1(dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.rc = 1.0 # number of right choice
        self.wc = 1.0 # number of wrong choice
        self.embedding = LocallyLinearEmbedding(n_components=(n_arm-1))
        self.lr = lr
    
    
    def update(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)
        
        
    def gradient_feature(self, g_list):
        new_g = []
        for g in g_list:
            s = np.array([sum(g), np.mean(g)])
            new_g.append(s)
        return np.array(new_g)

    def output_and_gradient(self, context):
        tensor = torch.from_numpy(context).float().to(device)
        res = self.func(tensor)
        g_list = []
        res_list = []
        for fx in res:
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            g_list.append(np.array(g.cpu()))
            res_list.append([fx.item()])
        res_list = np.array(res_list)
        g_list = np.array(g_list)
        g_list = self.embedding.fit_transform(g_list)
        #g_list = self.gradient_feature(g_list)
        return res_list, g_list
    
    
    def train(self, t):
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
                optimizer.zero_grad()
                loss = (self.func(c.to(device)) - r)**2
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000 * (int(t/1000)+1):
                    return tot_loss / cnt
            if batch_loss / length <= 1e-3:
                return batch_loss / length

            
class Network_2(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_2, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

    
class Network_2_fun:
    def __init__(self, dim, hidden=10, lr=0.01):
        self.func = Network_2(dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.lr = lr
        #print("network2:", hidden, self.lr)
    
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
                #delta = (self.func(c.to(device)) - r)*(r + 0.1)
                delta = self.func(c.to(device)) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 2000:
                    return tot_loss / cnt
            if batch_loss / length <= 1e-3:
                return batch_loss / length     
            
            
class Network_3(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_3, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x_1 = self.activate(self.activate(self.fc1(x)))
        return self.fc2(x_1)

class Network_3_fun:
    def __init__(self, dim, hidden=10, lr = 0.01):
        self.func = Network_3(dim, hidden_size=hidden).to(device)
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
                #loss = self.loss(output, target)
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

def relu(a, b):
    if b<0:
        return a
    if  a>b:
        return a-b
    else:
        return 0

def abs(a, b):
    if  a>b:
        return a-b
    else:
        return b-a        
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EE-Net')
    parser.add_argument('--dataset', default='mnist', type=str,  help='yelp, disin, mnist, movielens')
    args = parser.parse_args()
    regrets_all = []
    add_l1 = 1
    lr_1 = 0.0001
    lr_2 = 0.001
    lr_3 = 0.001
    for i in range(5):
        if args.dataset == "yelp":
            b = load_yelp()
            add_l1 = 1
        elif args.dataset == "disin":
            b = load_disin()
            lr_1 = 0.0001
            lr_2 = 0.001
            lr_3 = 0.001    
        elif args.dataset == "mnist":
            b = load_mnist_1d()
            lr_1 = 0.0001
            lr_2 = 0.001
            lr_3 = 0.001

        elif args.dataset == "movielens":
            b = load_movielen()
        else:
            b = Bandit_multi(dataset, 1, 0)

        regrets = []
        summ = 0
        l_1 = Network_1_fun(b.dim, b.n_arm, lr_1)
        l_2 = Network_2_fun(b.n_arm-1, 100, lr_2)
        l_3 = Network_3_fun(2, 20, lr_3)
        for t in range(10000):
            context, rwd = b.step()
            res1_list, gra_list = l_1.output_and_gradient(context)
            res2_list = l_2.output(gra_list)

            t_a = np.expand_dims([1/ np.log(t+150)]*b.n_arm, axis = 1)
            new_context = np.concatenate((res1_list, res2_list), axis=1)

            if t < 500:
                suml = res1_list + res2_list
                arm_select = np.argmax(suml)
            else:
                arm_select = l_3.select(new_context)


            r_1 = rwd[arm_select]

            l_1.update(context[arm_select], r_1)
            if r_1 == 1:
                index = 0
                for i in context:
                    if index != arm_select:
                        l_1.update(i, 0)
                    index += 1

            f_1 = res1_list[arm_select][0]
            #r_2 = r_1 - f_1

            def relu(x):
                if x > 0: return x
                else: return -x


            r_2 = r_1 - f_1
            l_2.update(gra_list[arm_select], r_2)
            r_3 = float(r_1)
            l_3.update(new_context[arm_select], r_3)
            if t < 10000:
                if r_1 == 0:
                    index = 0
                    for i in gra_list:
                        c = (1/np.log(t+10))
                        if index != arm_select:
                            l_2.update(i, c)
                        index += 1
            if t < 2000:
                if add_l1 and r_3 == 0.0:
                    index = 0
                    for i in new_context:
                        c = (1/np.log(t+10))
                        if index != arm_select:
                            l_3.update(i, c)
                        index += 1

            reg = np.max(rwd) - r_1
            gt = np.argmax(rwd)
            summ+=reg
            if t<1000:
                if t%10 == 0:
                    loss_1 = l_1.train(t)
                    loss_2 = l_2.train(t)
                    loss_3 = l_3.train(t)
            else:
                if t%100 == 0:
                    loss_1 = l_1.train(t)
                    loss_2 = l_2.train(t)
                    loss_3 = l_3.train(t)

            regrets.append(summ)
            if t % 50 == 0:
                print('round:{}, regret: {:},  average_regret: {:.3f}, loss_1:{:.4f}, loss_2:{:.4f}, loss_3:{:.4f}'.format(t,summ, summ/(t+1), loss_1, loss_2, loss_3))
        print(' regret: {:},  average_regret: {:.2f}'.format(summ, summ/(t+1)))
        regrets_all.append(regrets)



