import numpy as np
import argparse
import pickle
import os
import time
import torch
import pandas as pd 
import scipy as sp
import torch.nn as nn
import torch.optim as optim
import random
from collections import defaultdict
import scipy.io as sio
import scipy.sparse as spp
import scipy as sp


if torch.cuda.is_available():  
    dev = "cuda:1" 
else:  
    dev = "cpu" 
device = torch.device(dev)

