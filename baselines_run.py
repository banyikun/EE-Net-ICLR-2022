from KernelUCB import KernelUCB
from LinUCB import Linearucb
from Neural_epsilon import Neural_epsilon
from NeuralTS import NeuralTS
from NeuralUCB import NeuralUCBDiag
from NeuralNoExplore import NeuralNoExplore
import argparse
import numpy as np
import sys 

from load_data import load_yelp, load_mnist_1d, load_movielen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run baselines')
    parser.add_argument('--dataset', default='mnist', type=str, help='mnist, yelp, movielens, disin')
    parser.add_argument("--method", nargs="+", default=["Neural_epsilon", "NeuralTS", "NeuralUCB", "NeuralNoExplore"], help='list: ["KernelUCB", "LinUCB", "Neural_epsilon", "NeuralTS", "NeuralUCB", "NeuralNoExplore"]')
    parser.add_argument('--lamdba', default='0.1', type=float, help='Regulization Parameter')
    parser.add_argument('--nu', default='0.001', type=float, help='Exploration Parameter')
    
    args = parser.parse_args()
    dataset = args.dataset
    arg_lambda = args.lamdba 
    arg_nu = args.nu
    
    print("running methods:", args.method)
    for method in args.method:

        regrets_all = []
        for i in range(5):
            
            b = load_mnist_1d()
            
            if method == "KernelUCB":
                model = KernelUCB(b.dim, arg_lambda, arg_nu)

            elif method == "LinUCB":
                model = Linearucb(b.dim, arg_lambda, arg_nu)

            elif method == "Neural_epsilon":
                epsilon = 0.01
                model = Neural_epsilon(b.dim, epsilon)

            elif method == "NeuralTS":
                model = NeuralTS(b.dim, b.n_arm, m = 100, sigma = arg_lambda, nu = arg_nu)

            elif method == "NeuralUCB":
                model = NeuralUCBDiag(b.dim, lamdba = arg_lambda, nu = arg_nu,  hidden = 100)
                
            elif method == "NeuralNoExplore":
                model = NeuralNoExplore(b.dim)
            else:
                print("method is not defined. --help")
                sys.exit()
                
            regrets = []
            sum_regret = 0
            print("Round; Regret; Regret/Round")
            for t in range(10000):
                '''Draw input sample'''
                context, rwd = b.step()
                arm_select = model.select(context)
                reward = rwd[arm_select]

                if method == "LinUCB" or method == "KernelUCB":
                    model.train(context[arm_select],reward)

                elif method == "Neural_epsilon" or method == "NeuralUCB" or method == "NeuralTS" or method == "NeuralNoExplore":
                    model.update(context[arm_select], reward)
                    if t<1000:
                        if t%10 == 0:
                            loss = model.train(t)
                    else:
                        if t%100 == 0:
                            loss = model.train(t)

                regret = np.max(rwd) - reward
                sum_regret+=regret
                regrets.append(sum_regret)
                if t % 50 == 0:
                    print('{}: {:}, {:.4f}'.format(t, sum_regret, sum_regret/(t+1)))

            print("run:", i, "; ", "regret:", sum_regret)
            regrets_all.append(regrets)
        np.save("./results/{}_regret".format(method), regrets_all)
    
    
    
    
    
    
    
    
    
    
    
    
        