import torch
import torch.nn as nn

class MeanSquaredError(nn.Module):
    def __init__(self, dim=1):
        """
        Args:
            dim <int> or <tuple<int>>
        """
        super().__init__()
        
        self.dim = dim
        
        self.maximize = False
    
    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, *):
            target (batch_size, *):
        """
        loss = (input - target)**2 # (batch_size, *)
        loss = torch.mean(loss, dim=self.dim)
        
        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss
