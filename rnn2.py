import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

X = torch.arange(10).reshape((2, 5))
print(X.T)
Y=F.one_hot(X.T, 28)
print(Y)
