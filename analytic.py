#import numpy as np
import torch

def analytic_solution(x):
    sol =  (1 / (torch.exp(torch.pi*torch.ones(1)) - torch.exp(-torch.pi*torch.ones(1)))) * \
           torch.sin(torch.pi * x[0]) * (torch.exp(torch.pi * x[1]) - torch.exp(-torch.pi * x[1]))
    return sol