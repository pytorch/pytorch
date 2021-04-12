from .optimizer import Optimizer, _params_t

class SGD(Optimizer):
    def __init__(self, params: _params_t, lr: float, momentum: float=..., dampening: float=..., weight_decay:float=..., nesterov:bool=...) -> None: ...
