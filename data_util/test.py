import torch
import numpy as np



a = np.random.randn( 768)

print(torch.from_numpy(a).shape)