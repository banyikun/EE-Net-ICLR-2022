from EE_Net import Exploitation, Exploration, Decision_maker
from KernelUCB import KernelUCB
from LinUCB import Linearucb
from Neural_epsilon import Neural_epsilon
from NeuralTS import NeuralTS
from NeuralUCB import NeuralUCBDiag
import argparse
import numpy as np
import sys 

from load_data import load_yelp, load_mnist_1d, load_movielen, load_disin


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EE-Net')
    parser.add_argument('--dataset', default='mnist', type=str, help='mnist, yelp, movielens, disin')
    parser.add_argument('--method', default='NeuralUCB', type=str, help='EE-Net, KernelUCB, LinUCB, Neural_epsilon, NeuralTS, NeuralUCB')
    args = parser.parse_args()
    
    dataset = args.dataset
    method = args.method
    
    if dataset == "yelp":
        b = load_yelp()
        
    elif dataset == "disin":
        b = load_disin()
        
    elif dataset == "mnist":
        b = load_mnist_1d()

    elif dataset == "movielens":
        b = load_movielen()
    else:
        print("dataset is not defined. --help")
        sys.exit()
     
     

    elif method == "KernelUCB":
        arg_lambda = 0.1
        arg_nu= 0.0001
        model = KernelUCB(b.dim, arg_lambda, arg_nu)

    elif method == "LinUCB":
        arg_lambda = 0.1
        arg_nu= 0.001
        model = Linearucb(b.dim, arg_lambda, arg_nu)

    elif method == "Neural_epsilon":
        model = Neural_epsilon(b.dim, 0.1)

    elif method == "NeuralTS":
        sigma = 1
        nu = 0.01
        model = NeuralTS(b.dim, b.n_arm, m = 100, sigma = sigma, nu = nu)

    elif method == "NeuralUCB":
        arg_lambda = 0.01
        arg_nu = 1
        model = NeuralUCBDiag(b.dim, arg_lambda, arg_nu, 100)

    else:
        print("method is not defined. --help")
        sys.exit()
        
        
        
    regrets = []
    summ = 0
    print("Round; Regret; Regret/Round")
    for t in range(10000):
        '''Draw input sample'''
        context, rwd = b.step()
   
        if method == "LinUCB" or method == "KernelUCB":
            arm_select = model.select(context)
            r = rwd[arm_select]
            model.train(context[arm_select],r)

        elif method == "Neural_epsilon" or method == "NeuralUCB" or method == "NeuralTS":
            arm_select = model.select(context)
            r = rwd[arm_select]
            model.update(context[arm_select], r)
            if t<1000:
                if t%10 == 0:
                    loss = model.train(t)
            else:
                if t%100 == 0:
                    loss = model.train(t)
     
        r = rwd[arm_select]
        reg = np.max(rwd) - r
        summ+=reg
        regrets.append(summ)
        
        if t % 50 == 0:
            print('{}: {:}, {:.4f}'.format(t, summ, summ/(t+1)))
    
    
    print("round:", t, "; ", "regret:", summ)
    np.save("./regret",  regrets)
    
    
    
    
    
    
    
    
    
    
    
    
        