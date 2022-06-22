
# EE-Net

@inproceedings{ban2022eenet,

title={{EE}-Net: Exploitation-Exploration Neural Networks in Contextual Bandits},

author={Yikun Ban and Yuchen Yan and Arindam Banerjee and Jingrui He},

booktitle={International Conference on Learning Representations},

year={2022}
}


## Prerequisites: 

python 3.8.8

CUDA 11.2

torch 1.9.0

torchvision 0.10.0

sklearn 0.24.1

numpy 1.20.1

scipy 1.6.2

pandas 1.2.4




* EENet_run.py  - Run proposed algorithm 
* Neural_epsilon.py - fully-connected neural network with epsilon-greedy exploration strategy
* NeuralTS.py - Neural thompson sampling  [Zhang et al. 2020]
* NeuralUCB.py - NeuralUCB [Zhou et al. 2020]
* KernelUCB.py - KernelUCB [Valko et al., 2013a]
* LinUCB.py - LinUCB [Li et al., 2010]

* packages.py - all the needed packages
* load_data.py - load the datasets
* movie_10000items_2000users_feature.npy - processed movielens data
* yelp_10000items_2000users_features.npy - processed yelp data


For disin dataset, as the processed files are too large, feel free to contact me if you need them. 

## Methods:
EE-Net, KernelUCB, LinUCB, Neural_epsilon, NeuralTS, NeuralUCB 


## Datasets:
mnist, yelp, disin, movielens


## Run:
python run.py --dataset "dataset" --method "method"

For example, python run.py --dataset mnist --method EE-Net   