import numpy as np
import matplotlib.pyplot as plt


def get_mean_std(ress):
    return np.mean(ress, axis=0), np.std(ress, axis =0)
    

if __name__ == '__main__':    
    x = range(10000)
    plt.figure(figsize=(10, 6))


    ee = np.load('./results/eenet_results1.npy')
    ee_mean, ee_std = get_mean_std(ee)
    plt.plot(x, ee_mean, 'k-', color='red',linewidth=2.0,linestyle='-', label = 'EE-Net')
    plt.fill_between(x, ee_mean-ee_std, ee_mean+ee_std, facecolor='red', alpha=0.2)

    
    ucb = np.load("./results/NeuralUCB_regret.npy")
    ucb_mean, ucb_std = get_mean_std(ucb)
    plt.plot(x, ucb_mean, 'k-', color='blue',linewidth=2.0,linestyle=':', label = 'NeuralUCB')
    plt.fill_between(x, ucb_mean-ucb_std, ucb_mean+ucb_std, facecolor='blue', alpha=0.2)

    ts = np.load("./results/NeuralTS_regret.npy")
    ts_mean, ts_std = get_mean_std(ts)
    plt.plot(x, ts_mean, 'k-', color='yellow',linewidth=2.0,linestyle=':', label = 'NeuralTS')
    plt.fill_between(x, ts_mean-ts_std, ts_mean+ts_std, facecolor='green', alpha=0.2)

    ep = np.load("./results/Neural_epsilon_regret.npy")
    ep_mean, ep_std = get_mean_std(ep)
    plt.plot(x, ep_mean, 'k-', color='green',linewidth=2.0,linestyle='-.', label = "NeuralEpsilon")
    plt.fill_between(x, ep_mean-ep_std, ep_mean+ep_std, facecolor='orange', alpha=0.2)

    plt.xlabel('Rounds')
    plt.ylabel('Regret')
    plt.legend()
    plt.title("Mnist")
    #plt.rcParams["figure.figsize"] = (20, 10)
    plt.savefig('./figures/regret_mnist.pdf', dpi=500)