
import numpy as np
import torch
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../contact_pred/'))
from structure import compute_kabsch_RMSD

a = torch.Tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0], [1, 0, 0], [2, 0, 0]]])
b = torch.Tensor([[[0, 0, 0], [999, 0, 0], [2, 0, 0]], [[0, 0, 0], [0, 3, 0], [0, 6, 0]]])
w = torch.Tensor([[1, 0, 1], [1, 1, 1]])
#print(a, b, a.shape, b.shape)
#print(w, w.shape)

print(compute_kabsch_RMSD(a, b, weight=w))
