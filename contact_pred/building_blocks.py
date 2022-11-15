import torch
import torch.nn as nn

class MLP(torch.nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self, n, nlayers):
        super().__init__()
        hidden_layers = [(torch.nn.Linear(n, n),torch.nn.GELU())
            for _ in range(nlayers-1)]
        self.layers = torch.nn.Sequential(
               *[item for layer_pair in hidden_layers for item in layer_pair],
               torch.nn.Linear(n,n),
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)
