import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# 去噪神经网络
class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device, t_dim):
        super(MLP, self).__init__()
        
        self.t_dim = t_dim
        self.action_dim = action_dim
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            
        )