from .optimizer import Optimizer
from typing import Any, Callable, TypeAlias, Iterator
from . import Muon, Adam

StateDict: TypeAlias = dict[str, Any]



class OptimizerDict(Optimizer):
    def __init__(self, optimizer_dict: dict[str, Optimizer]) -> None:
        for optimizer in optimizer_dict.values():
            if not isinstance(optimizer, Optimizer):
                raise ValueError(f"{optimizer} is not a valid optimizer")
        self._optimizer_dict = optimizer_dict

    def state_dict(self) -> StateDict:
        state_dict = {}
        for optimizer_name, optimizer in self._optimizer_dict.items():
            state_dict[optimizer_name] = optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: StateDict) -> None:
        for optimizer_name, optimizer_state in state_dict.items():
            if optimizer_name not in self._optimizer_dict:
                raise ValueError(f"{optimizer_name} is not a valid optimizer")
            self._optimizer_dict[optimizer_name].load_state_dict(optimizer_state)

    def step(self, closure: dict[Callable[[], float]] = None) -> None:
        '''
        Steps each optimizer in the optimizer dictionary.
        However, handling the return values of the optimizers is not currently supported.
        We could return a dictionary containing the return values of the optimizers, 
        but optimizer.step() is typically expected to return None or a tensor

        '''
        for key in self._optimizer_dict.keys():
            optimizer = self._optimizer_dict[key]
            if closure is not None: 
                optimizer_closure = closure[key]
            else:
                optimizer_closure = None
            optimizer.step(optimizer_closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self._optimizer_dict.values():
            optimizer.zero_grad(set_to_none)
        
    def __getitem__(self, key: str) -> Optimizer:
        return self._optimizer_dict[key]

    def __repr__(self) -> str:
        return f"OptimizerStateDict({self._optimizer_dict})"

    def __len__(self) -> int:
        return len(self._optimizer_dict)

    def __iter__(self) -> Iterator[Optimizer]:
        return iter(self._optimizer_dict)

    def keys(self):
        return self._optimizer_dict.keys()

def MakeMuonWithFallback(params, fallback_optimizer_type, muon_kwargs, fallback_kwargs):
    muon_params = []
    fallback_params = []
    for param in params:
        if param.ndim == 2:
            muon_params.append(param)
        else:
            fallback_params.append(param)
    muon = Muon(muon_params, **muon_kwargs)
    fallback_optimizer = fallback_optimizer_type(fallback_params, **fallback_kwargs)
    return OptimizerDict({"Muon": muon, f"{fallback_optimizer_type.__name__}": fallback_optimizer})