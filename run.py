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
    parser = argparse.ArgumentParser(description='Meta-Ban')
    parser.add_argument('--dataset', default='mnist', type=str, help='mnist, yelp, movielens, disin')
    parser.add_argument('--method', default='EE-Net', type=str, help='EE-Net, KernelUCB, LinUCB, Neural_epsilon, NeuralTS, NeuralUCB')
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
     
     
    if method == "EE-Net":   
        if dataset == "yelp":
            lr_1 = 0.001
            lr_2 = 0.01
            lr_3 = 0.01  
        elif dataset == "disin":
            lr_1 = 0.0005
            lr_2 = 0.0001
            lr_3 = 0.001    
        elif dataset == "mnist":
            lr_1 = 0.01
            lr_2 = 0.01
            lr_3 = 0.01  
        elif dataset == "movielens":
            lr_1 = 0.001
            lr_2 = 0.001
            lr_3 = 0.001  

        f_1 = Exploitation(b.dim, b.n_arm, lr_1)
        f_2 = Exploration(b.n_arm-1, 100, lr_2)
        f_3 = Decision_maker(2, 20, lr_3)    
        

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
        arg_nu = 0.1
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
        
        if method == "EE-Net":   
            '''exploitation score and embedded gradient'''
            res1_list, gra_list = f_1.output_and_gradient(context)

            '''exploration score'''
            res2_list = f_2.output(gra_list)

            '''build input for decision maker'''
            new_context = np.concatenate((res1_list, res2_list), axis=1)

            '''hybrid decision maker'''
            if t < 500:
                '''sample linear model'''
                sum_list12 = res1_list + res2_list
                arm_select = np.argmax(sum_list12)
            else:
                '''neural model'''
                arm_select = f_3.select(new_context)

            '''reward'''
            r_1 = rwd[arm_select]

            f_1.update(context[arm_select], r_1)
            f_1_score = res1_list[arm_select][0]

            '''label for exploration network'''
            r_2 = r_1 - f_1_score
            f_2.update(gra_list[arm_select], r_2)

            '''creat additional samples for exploration network'''
            if r_1 == 0:
                index = 0
                for i in gra_list:
                    '''set small scores for un-selected arms if the selected arm is 0-reward'''
                    small_score = (1/np.log(t+10))
                    if index != arm_select:
                        f_2.update(i, small_score)
                    index += 1

            '''label for decision maker'''
            r_3 = float(r_1)
            f_3.update(new_context[arm_select], r_3)
            
             
            '''training'''
            if t<1000:
                if t%10 == 0:
                    loss_1 = f_1.train(t)
                    loss_2 = f_2.train(t)
                    loss_3 = f_3.train(t)
            else:
                if t%100 == 0:
                    loss_1 = f_1.train(t)
                    loss_2 = f_2.train(t)
                    loss_3 = f_3.train(t)

        elif method == "LinUCB" or method == "KernelUCB":
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
    
    
    
    
    
    
    
    
    
    
    
    
        
