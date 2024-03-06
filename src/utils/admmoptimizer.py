# Description: ADMM optimizer
import torch
from torch.optim import Optimizer

class ADMMOpt(Optimizer):
    '''Implements the ADMM optimizer'''
    ' below is arbritrary'

    def __init__(self, params, rho=0.01, lamb=0.01, zeta=0.01):
        defaults = dict(rho=rho, lamb=lamb, zeta=zeta)
        super(ADMMOpt, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            rho = group['rho']
            lamb = group['lamb']
            zeta = group['zeta']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data = (d_p + rho * (p.data - zeta)) / (1 + rho)
                zeta = zeta + lamb * (p.data - d_p)

        return loss
    
