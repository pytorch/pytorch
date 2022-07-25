import torch
from .grad_scaler import GradScaler
class OverlappedGradScaler(GradScaler):
    def __init__(self, 
                 init_scale=2 ** 16, 
                 growth_factor=2, 
                 backoff_factor=0.5, 
                 growth_interval=2000, 
                 enabled=True):
        super().__init__(init_scale, 
                         growth_factor, 
                         backoff_factor, 
                         growth_interval, 
                         enabled)
    def _unscale_grads_(self, param, inv_scale, found_inf, allow_fp16):
        with torch.no_grad:
            if param.grad is None:
                return
            if (not allow_fp16) and param.grad.dtype == torch.float16:
                raise ValueError("Attempting to unscale FP16 gradients.")
            if param.grad.is_sparse:
                # is_coalesced() == False means the sparse grad has values with duplicate indices.
                # coalesce() deduplicates indices and adds all values that have the same index.
                # For scaled fp16 values, there's a good chance coalescing will cause overflow,
                # so we should check the coalesced _values().
                if param.grad.dtype is torch.float16:
                    param.grad = param.grad.coalesce()
                to_unscale = param.grad._values()
            else:
                to_unscale = param.grad
        torch._amp_foreach_non_finite_check_and_unscale_(to_unscale,
                                                         found_inf,
                                                         inv_scale)
    
    def unscale_(self, param) -> bool:
        """
        Unscale the grad.
        Return True if no inf founded.
        """
        if not self._enabled:
            return
        fix = "This may indicate your script did not use scaler.scale(loss or outputs) earlier in the iteration."
        assert self._scale is not None, "Attempted unscale_ but _scale is None. " + fix
        inv_scale = self._scale.double().reciprocal().float()
        found_inf = torch.full((1,), 0.0, dtype=torch.float32, device=self._scale.device)
        self._unscale_grads_(param, inv_scale, found_inf, False)
        return found_inf.item() == 0
        
    def step(self, optimizer, *args, **kwargs):
        """Disable ``step``"""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support method step."
        )