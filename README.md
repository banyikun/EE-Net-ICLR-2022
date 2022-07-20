
# EE-Net: Exploitation-Exploration Neural Networks in Contextual Bandits

In this repository, we provide one implementation of EE-Net, where the linear decision-maker (e.g., f1 + f2 ) and neural decision-maker (f3) are provided respectively. For the exploration network, one-layer CNN is used to reduce the dimensionality of gradient of exploitation network. 


## Run:

Run EE-Net on Mnist:

```bash
python EENet_run.py
```

Run baselines on Mnist:

```bash
python  baselines/baselines_run.py
```



## Prerequisites: 

python 3.8.8, CUDA 11.2, torch 1.9.0, torchvision 0.10.0, sklearn 0.24.1, numpy 1.20.1, scipy 1.6.2, pandas 1.2.4


## Hyper-parameters

[dim](): dimensionality of arm context vector

[n_arm](): number of arms.

[pooling_step_size](): aggregation size for the gradient, and the aggregated gradient will be the input of f2

[hidden](): width of all neural networks
