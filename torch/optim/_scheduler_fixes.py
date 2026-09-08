import torch

from . import lr_scheduler


_original_constant_lr_init = lr_scheduler.ConstantLR.__init__
_original_linear_lr_init = lr_scheduler.LinearLR.__init__
_original_reduce_lr = lr_scheduler.ReduceLROnPlateau._reduce_lr


def _constant_lr_init(self, optimizer, factor=1.0 / 3, total_iters=5, last_epoch=-1):
    if factor > 1.0 or factor < 0:
        raise ValueError("Constant multiplicative factor expected to be between 0 and 1.")
    if total_iters < 0:
        raise ValueError("total_iters must be non-negative")
    _original_constant_lr_init(self, optimizer, factor, total_iters, last_epoch)


def _linear_lr_init(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1):
    if start_factor > 1.0 or start_factor <= 0:
        raise ValueError(
            "Starting multiplicative factor expected to be greater than 0 and less or equal to 1."
        )
    if end_factor > 1.0 or end_factor < 0:
        raise ValueError("Ending multiplicative factor expected to be between 0 and 1.")
    if total_iters <= 0:
        raise ValueError("total_iters must be greater than 0")
    _original_linear_lr_init(
        self, optimizer, start_factor, end_factor, total_iters, last_epoch
    )


def _reduce_lr(self, epoch):
    if len(self.optimizer.param_groups) != len(self.min_lrs):
        if self.default_min_lr is None:
            raise RuntimeError(
                "The number of param groups in the `optimizer` "
                f"({len(self.optimizer.param_groups)}) differs "
                f"from when `ReduceLROnPlateau` was initialized "
                f"({len(self.min_lrs)}), usually due to a new "
                "param group being added to the optimizer. Please "
                "modify the `min_lrs` field to match the length "
                "of the `optimizer` param groups."
            )
        self.min_lrs = [self.default_min_lr] * len(self.optimizer.param_groups)

    for i, param_group in enumerate(self.optimizer.param_groups):
        old_lr = param_group["lr"]
        if isinstance(old_lr, torch.Tensor):
            min_lr = torch.as_tensor(
                self.min_lrs[i], device=old_lr.device, dtype=old_lr.dtype
            )
            new_lr = torch.maximum(old_lr * self.factor, min_lr)
            if bool(torch.any(old_lr - new_lr > self.eps)):
                lr_scheduler._update_param_group_val(param_group, "lr", new_lr)
        else:
            _original_reduce_lr(self, epoch)
            break


def _composite_initial_step(self):
    self._step_count = 0
    self._last_lr = lr_scheduler._param_groups_val_list(self.optimizer, "lr")


def install() -> None:
    lr_scheduler.ConstantLR.__init__ = _constant_lr_init
    lr_scheduler.LinearLR.__init__ = _linear_lr_init
    lr_scheduler.ReduceLROnPlateau._reduce_lr = _reduce_lr
    lr_scheduler.SequentialLR._initial_step = _composite_initial_step
    lr_scheduler.ChainedScheduler._initial_step = _composite_initial_step
