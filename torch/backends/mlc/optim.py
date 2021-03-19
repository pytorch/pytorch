from .mlc_adam import MLCAdam
from .mlc_rmsprop import MLCRMSprop
from .mlc_sgd import MLCSGD
import torch
setattr(torch.mlc, 'MLCAdam', MLCAdam)
setattr(torch.mlc, 'MLCSGD', MLCSGD)
setattr(torch.mlc, 'MLCRMSprop', MLCRMSprop)
setattr(torch.optim, 'MLCAdam', MLCAdam)
setattr(torch.optim, 'MLCSGD', MLCSGD)
setattr(torch.optim, 'MLCRMSprop', MLCRMSprop)
