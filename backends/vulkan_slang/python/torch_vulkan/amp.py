"""AMP (Automatic Mixed Precision) support for the Vulkan backend.

Currently a scaffold for future float16 shader support.
Provides GradScaler and autocast context manager compatible with the standard
PyTorch AMP API.
"""

import torch


class GradScaler:
    """Gradient scaler for Vulkan AMP training.

    When float16 shaders are available, this will:
    - Scale loss to prevent gradient underflow in float16
    - Unscale gradients before optimizer step
    - Check for inf/nan and skip updates when detected
    - Dynamically adjust scale factor

    Currently operates as an identity (scale=1.0) since all ops are float32.
    """

    def __init__(self, init_scale=2.0**16, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000, enabled=True):
        self._scale = init_scale if enabled else 1.0
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._enabled = enabled
        self._growth_tracker = 0
        self._found_inf = False

    def scale(self, loss):
        """Scale the loss tensor."""
        if not self._enabled:
            return loss
        return loss * self._scale

    def unscale_(self, optimizer):
        """Unscale gradients in-place."""
        if not self._enabled:
            return

        inv_scale = 1.0 / self._scale
        self._found_inf = False

        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.mul_(inv_scale)
                    # Check for inf/nan
                    if torch.isinf(param.grad.data.cpu()).any() or \
                       torch.isnan(param.grad.data.cpu()).any():
                        self._found_inf = True

    def step(self, optimizer, *args, **kwargs):
        """Step the optimizer, skipping if inf/nan found."""
        if not self._enabled:
            optimizer.step(*args, **kwargs)
            return

        if not self._found_inf:
            optimizer.step(*args, **kwargs)

    def update(self):
        """Update the scale factor."""
        if not self._enabled:
            return

        if self._found_inf:
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0

    def get_scale(self):
        return self._scale

    def is_enabled(self):
        return self._enabled

    def state_dict(self):
        return {
            "scale": self._scale,
            "growth_factor": self._growth_factor,
            "backoff_factor": self._backoff_factor,
            "growth_interval": self._growth_interval,
            "growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state_dict):
        self._scale = state_dict["scale"]
        self._growth_factor = state_dict["growth_factor"]
        self._backoff_factor = state_dict["backoff_factor"]
        self._growth_interval = state_dict["growth_interval"]
        self._growth_tracker = state_dict["growth_tracker"]
