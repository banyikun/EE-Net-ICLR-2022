from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd 

import torch
import torchvision
from torchvision import datasets, transforms



class Bandit_multi:
    def __init__(self, name, is_shuffle=True, seed=None):
        # Fetch data
        if name == 'mnist':
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
            X = X.to_numpy()
            y = y.to_numpy()
            print('mnist', X.shape)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'covertype':
            X, y = fetch_openml('covertype', version=3, return_X_y=True)
            X = X.to_numpy()
            y = y.to_numpy()
            # avoid nan, set nan as -1
            X[pd.isnull(X)] = - 1
            X = normalize(X)
        elif name == 'MagicTelescope':
            X, y = fetch_openml('MagicTelescope', version=1, return_X_y=True)
            X = X.to_numpy()
            y = y.to_numpy()
            print('MagicTelescope', X.shape)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'shuttle':
            X, y = fetch_openml('shuttle', version=1, return_X_y=True)
            X = X.to_numpy()
            y = y.to_numpy()
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'miceprotein':
            X, y = fetch_openml('miceprotein', version=4, return_X_y=True)
            X = X.to_numpy()
            y = y.to_numpy()
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'optdigits':
            X, y = fetch_openml('optdigits', return_X_y=True)
            X = X.to_numpy()
            y = y.to_numpy()
            # avoid nan, set nan as -1
            X[pd.isnull(X)] = - 1
            X = normalize(X)
        elif name == 'notmnist':
            X = np.load('./nomnist/imagedat.npy', allow_pickle=True)
            y = np.load('./nomnist/labeldata.npy', allow_pickle=True)
            new_X = []
            for i in X:
                i = i.flatten()
                new_X.append(i)
            X = np.array(new_X)
            print('notmnist', X.shape)
            X[np.isnan(X)] = - 1
            X = normalize(X)
        else:
            raise RuntimeError('Dataset does not exist')
        # Shuffle data
        if is_shuffle:
            self.X, self.y = shuffle(X, y, random_state=seed)
        else:
            self.X, self.y = X, y
        # generate one_hot coding:
        self.y_arm = OrdinalEncoder(
            dtype=np.int).fit_transform(self.y.reshape((-1, 1)))
        # cursor and other variables
        self.cursor = 0
        self.size = self.y.shape[0]
        self.n_arm = np.max(self.y_arm) + 1
        self.dim = self.X.shape[1] * self.n_arm
        self.act_dim = self.X.shape[1]

    def step(self):
        assert self.cursor < self.size
        X = np.zeros((self.n_arm, self.dim))
        for a in range(self.n_arm):
            X[a, a * self.act_dim:a * self.act_dim +
                self.act_dim] = self.X[self.cursor]
        arm = self.y_arm[self.cursor][0]
        rwd = np.zeros((self.n_arm,))
        rwd[arm] = 1
        self.cursor += 1
        return X, rwd

    def finish(self):
        return self.cursor == self.size

    def reset(self):
        self.cursor = 0


class load_mnist_1d:
    def __init__(self):
        # Fetch data
        batch_size = 1
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset1 = datasets.MNIST('./data', train=True, download=True,
                   transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size,
                                      shuffle=True, num_workers=2)
        self.dataiter = iter(train_loader)
        self.n_arm = 10
        self.dim = 7840
 
    def step(self):
        x, y = self.dataiter.next()
        d = x.numpy()[0]
        d = d.reshape(784)
        target = y.item()
        X_n = []
        for i in range(10):
            front = np.zeros((784*i))
            back = np.zeros((784*(9 - i)))
            new_d = np.concatenate((front,  d, back), axis=0)
            X_n.append(new_d)
        X_n = np.array(X_n)    
        rwd = np.zeros(self.n_arm)
        #print(target)
        rwd[target] = 1
        return X_n, rwd  
    
    

class load_yelp:
    def __init__(self):
        # Fetch data
        self.m = np.load("./yelp_2000users_10000items_entry.npy")
        self.U = np.load("./yelp_2000users_10000items_features.npy")
        self.I = np.load("./yelp_10000items_2000users_features.npy")
        self.n_arm = 10
        self.dim = 20
        self.pos_index = []
        self.neg_index = []
        for i in self.m:
            if i[2] ==1:
                self.pos_index.append((i[0], i[1]))
            else:
                self.neg_index.append((i[0], i[1]))   
            
        self.p_d = len(self.pos_index)
        self.n_d = len(self.neg_index)
        print(self.p_d, self.n_d)
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)


    def step(self):        
        arm = np.random.choice(range(10))
        #print(pos_index.shape)
        pos = self.pos_index[np.random.choice(range(self.p_d), 9, replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), replace=False)]
        X_ind = np.concatenate((pos[:arm], [neg], pos[arm:]), axis=0)
        X = []
        for ind in X_ind:
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]])))
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        return np.array(X), rwd
    
    
    
class load_movielen:
    def __init__(self):
        # Fetch data
        self.m = np.load("./movie_2000users_10000items_entry.npy")
        self.U = np.load("./movie_2000users_10000items_features.npy")
        self.I = np.load("./movie_10000items_2000users_features.npy")
        self.n_arm = 10
        self.dim = 20
        self.pos_index = []
        self.neg_index = []
        for i in self.m:
            if i[2] ==1:
                self.pos_index.append((i[0], i[1]))
            else:
                self.neg_index.append((i[0], i[1]))   
            
        self.p_d = len(self.pos_index)
        self.n_d = len(self.neg_index)
        print(self.p_d, self.n_d)
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)


    def step(self):        
        arm = np.random.choice(range(10))
        #print(pos_index.shape)
        pos = self.pos_index[np.random.choice(range(self.p_d), 9, replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), replace=False)]
        X_ind = np.concatenate((pos[:arm], [neg], pos[arm:]), axis=0)
        X = []
        for ind in X_ind:
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]])))
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        return np.array(X), rwd
    
    
    
    
class load_disin:
    def __init__(self):
        # Fetch data
        self.d = np.load('./feature_matrix.pkl', allow_pickle=True)
        self.l = np.load('./label_matrix.pkl', allow_pickle=True)
        self.n_arm = 10
        self.dim = 300
        self.pos_index = []
        self.neg_index = []
        for i in range(len(self.l)):
            if self.l[i] > 0:
                self.pos_index.append(i)
            else: 
                self.neg_index.append(i)
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)
        print(len(self.pos_index), len(self.neg_index))


    def step(self):        
        arm = np.random.choice(range(10))
        #print(pos_index.shape)
        pos = np.random.choice(self.pos_index, size = 9, replace=False)
        neg = np.random.choice(self.neg_index, size = 1, replace=False)
        X_ind = np.concatenate((pos, neg))
        np.random.shuffle(X_ind)
        X = []
        rwd = []
        for ind in X_ind:
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(self.d[ind])
            if self.l[ind]>0:
                rwd.append(0.0)
            else:
                rwd.append(1.0)
        return np.array(X), rwd
    
class load_disin_20:
    def __init__(self):
        # Fetch data
        self.d = np.load('./feature_matrix.pkl', allow_pickle=True)
        self.l = np.load('./label_matrix.pkl', allow_pickle=True)
        self.n_arm = 20
        self.dim = 300
        self.pos_index = []
        self.neg_index = []
        for i in range(len(self.l)):
            if self.l[i] > 0:
                self.pos_index.append(i)
            else: 
                self.neg_index.append(i)
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)
        print(len(self.pos_index), len(self.neg_index))


    def step(self):        
        #print(pos_index.shape)
        pos = np.random.choice(self.pos_index, size = 19, replace=False)
        neg = np.random.choice(self.neg_index, size = 1, replace=False)
        X_ind = np.concatenate((pos, neg))
        np.random.shuffle(X_ind)
        X = []
        rwd = []
        for ind in X_ind:
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(self.d[ind])
            if self.l[ind]>0:
                rwd.append(0.0)
            else:
                rwd.append(1.0)
        return np.array(X), rwd