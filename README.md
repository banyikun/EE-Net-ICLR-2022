
# EE-Net: Exploitation-Exploration Neural Networks in Contextual Bandits

In this repository, we provide one implementation of EE-Net, where the linear decision-maker (e.g., f1 + f2 ) and neural decision-maker (f3) are provided respectively. For the exploration network, one-layer CNN is used to reduce the dimensionality of gradient of exploitation network. 


## Run:

Run EE-Net on Mnist:

```bash
python EENet_run.py
```

Run baselines on Mnist:

```bash
python baselines/baselines_run.py
```



## Prerequisites: 

python 3.8.8, CUDA 11.2, torch 1.9.0, torchvision 0.10.0, sklearn 0.24.1, numpy 1.20.1, scipy 1.6.2, pandas 1.2.4


## Hyper-parameters

[dim](https://github.com/banyikun/EE-Net-ICLR-2022/blob/d85d1f98b38d80ccb37f2f73cc964804f321fc68/EENet.py#L6): dimensionality of arm context vector

[n_arm](https://github.com/banyikun/EE-Net-ICLR-2022/blob/d85d1f98b38d80ccb37f2f73cc964804f321fc68/EENet.py#L6): number of arms.

[pooling_step_size](https://github.com/banyikun/EE-Net-ICLR-2022/blob/d85d1f98b38d80ccb37f2f73cc964804f321fc68/EENetClass.py#L96): aggregation size for the gradient, and the aggregated gradient will be the input of f2

[hidden](https://github.com/banyikun/EE-Net-ICLR-2022/blob/d85d1f98b38d80ccb37f2f73cc964804f321fc68/EENet.py#L6): width of all neural networks








If you use the codes of this repository, please kindly cite the following papers:

````
@inproceedings{ban2022eenet,
title={{EE}-Net: Exploitation-Exploration Neural Networks in Contextual Bandits},
author={Yikun Ban and Yuchen Yan and Arindam Banerjee and Jingrui He},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=X_ch3VrNSRg}
}

@article{ban2023neural,
  title={Neural exploitation and exploration of contextual bandits},
  author={Ban, Yikun and Yan, Yuchen and Banerjee, Arindam and He, Jingrui},
  journal={arXiv preprint arXiv:2305.03784},
  year={2023}
}
````

