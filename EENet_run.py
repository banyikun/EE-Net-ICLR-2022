from baselines.load_data import load_mnist_1d
from EENet import EE_Net
import numpy as np
import os



if __name__ == '__main__':
    dataset = 'mnist'
   
    runing_times = 3
    regrets_all = []
    b = load_mnist_1d()
    for i in range(runing_times):  

        lr_1 = 0.01 #learning rate for exploitation network
        lr_2 = 0.001 #learning rate for exploration network
        lr_3 = 0.001 #learning rate for decision maker

        regrets = []
        sum_regret = 0
        ee_net = EE_Net(b.dim, b.n_arm, pool_step_size = 50, lr_1 = lr_1, lr_2 = lr_2, lr_3 = lr_3,  hidden=100, neural_decision_maker = False)
        for t in range(10000):
            context, rwd = b.step()
            arm_select = ee_net.predict(context, t)

            reward = rwd[arm_select]
            regret = np.max(rwd) - reward

            ee_net.update(context, reward, t)

            sum_regret += regret
            if t<1000:
                if t%10 == 0:
                    loss_1, loss_2, loss_3  = ee_net.train(t)

            else:
                if t%100 == 0:
                    loss_1, loss_2, loss_3  = ee_net.train(t)

            regrets.append(sum_regret)
            if t % 50 == 0:
                print('round:{}, regret: {:},  average_regret: {:.3f}, loss_1:{:.4f}, loss_2:{:.4f}, loss_3:{:.4f}'.format(t,sum_regret, sum_regret/(t+1), loss_1, loss_2, loss_3))
        print(' regret: {:},  average_regret: {:.2f}'.format(sum_regret, sum_regret/(t+1)))
        regrets_all.append(regrets)
    path = os.getcwd()
    np.save('{}/results/eenet_results.npy'.format(path), regrets_all)