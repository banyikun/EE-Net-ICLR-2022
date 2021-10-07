
# EE-Net

## Prerequisites: 

python 3.8.8

CUDA 11.2

torch 1.9.0

torchvision 0.10.0

sklearn 0.24.1

numpy 1.20.1

scipy 1.6.2

pandas 1.2.4


## Methods:

* EE-Net.py  -  Proposed algorithm 
* neural-epsilon.py - fully-connected neural network with epsilon-greedy exploration strategy
* neuralTS.py - Neural thompson sampling  [Zhang et al. 2020]
* neuralUCB.py - Neural UCB [Zhou et al. 2020]
* kernalUCB.py - Kernal UCB [Valko et al., 2013a]
* linUCB.py - LinUCB [Li et al., 2010]
* neural-noexplore.py - pure exploitation with fully-connected neural network

* packages.py - all the needed packages
* load_data.py - load the datasets

## Datasets:
mnist, yelp, disin, movielens

## Run:
python "method" --dataset "dataset"

For example,   python EE-Net.py --dataset mnist   ; python neuralUCB.py --dataset yelp
