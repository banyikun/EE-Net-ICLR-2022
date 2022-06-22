import numpy as np
import matplotlib.pyplot as plt


def get_mean_std(ress):
    return np.mean(ress, axis=0), np.std(ress, axis =0)
    

if __name__ == '__main__':    
    x = range(10000)

    ee = np.load('./eenet_results.npy')
    ee_mean, ee_std = get_mean_std(ee)
    plt.plot(x, ee_mean, 'k-', color='red',linewidth=2.0,linestyle='-', label = 'EE-Net')
    plt.fill_between(x, ee_mean-ee_std, ee_mean+ee_std, facecolor='red', alpha=0.2)

    '''
    ucb = np.load("./data/fc_mnist_10000_5runs.npy")
    ucb_mean, ucb_std = get_mean_std(ucb)
    plt.plot(x, y_2, 'k-', color='blue',linewidth=2.0,linestyle='-', label = 'NeuralUCB')
    plt.fill_between(x, ucb_mean-fc_std, ucb_mean+fc_std, facecolor='blue', alpha=0.2)

    ts = np.load("./data/neuralTS_mnist_10000_5runs.npy")
    ts_mean, ts_std = get_mean_std(ts)
    plt.plot(x, y_3, 'k-', color='yellow',linewidth=2.0,linestyle='-', label = 'NeuralTS')
    plt.fill_between(x, ts_mean-ts_std, ts_mean+ts_std, facecolor='green', alpha=0.2)

    plt.plot(x, y_7, 'k-', color='green',linewidth=2.0,linestyle='-.', label = "KernalUCB")
    plt.fill_between(x, y_7-ker_std, y_7+ker_std, facecolor='orange', alpha=0.2)

    plt.plot(x, y_6, 'k-', color='orange',linewidth=2.0,linestyle='-.', label = "LinUCB")
    plt.fill_between(x, y_6-ep_std, y_6+ep_std, facecolor='orange', alpha=0.2)

    plt.plot(x, y_4, 'k-', color='grey',linewidth=2.0,linestyle='-', label = "Neural-epsilon")
    plt.fill_between(x, y_4-ep_std, y_4+ep_std, facecolor='grey', alpha=0.2)
    '''


    plt.xlabel('Rounds')
    plt.ylabel('Regret')
    plt.legend()
    plt.title("Mnist")
    plt.rcParams["figure.figsize"] = (7,4)
    plt.savefig('./figures/regret_mnist_new.pdf', dpi=500,bbox_inches = 'tight')