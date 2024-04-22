import torch


_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')