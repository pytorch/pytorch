import threading

import torch

from . import lr_scheduler


_original_lrscheduler_init = lr_scheduler.LRScheduler.__init__
_original_constant_lr_init = lr_scheduler.ConstantLR.__init__
_original_linear_lr_init = lr_scheduler.LinearLR.__init__
_original_step_lr_init = lr_scheduler.StepLR.__init__
_original_polynomial_lr_init = lr_scheduler.PolynomialLR.__init__
_original_cosine_lr_init = lr_scheduler.CosineAnnealingLR.__init__
_original_cyclic_lr_init = lr_scheduler.CyclicLR.__init__
_original_one_cycle_lr_init = lr_scheduler.OneCycleLR.__init__
_install_lock = threading.Lock()


def _lrscheduler_init(self, optimizer, last_epoch=-1):
    _original_lrscheduler_init(self, optimizer, last_epoch)
    self.base_lrs = [
        lr.clone() if isinstance(lr, torch.Tensor) else lr for lr in self.base_lrs
    ]


def _constant_lr_init(self, optimizer, factor=1.0 / 3, total_iters=5, last_epoch=-1):
    if factor > 1.0 or factor <= 0:
        raise ValueError(f"factor must be positive and at most 1, but got {factor}")
    if total_iters < 0:
        raise ValueError("total_iters must be non-negative")
    _original_constant_lr_init(self, optimizer, factor, total_iters, last_epoch)


def _linear_lr_init(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1):
    if start_factor > 1.0 or start_factor <= 0:
        raise ValueError(
            f"start_factor must be positive and at most 1, but got {start_factor}"
        )
    if end_factor > 1.0 or end_factor < 0:
        raise ValueError(
            f"end_factor must be between 0 and 1, but got {end_factor}"
        )
    if total_iters <= 0:
        raise ValueError(f"total_iters must be positive, but got {total_iters}")
    _original_linear_lr_init(
        self, optimizer, start_factor, end_factor, total_iters, last_epoch
    )


def _step_lr_init(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, but got {step_size}")
    _original_step_lr_init(self, optimizer, step_size, gamma, last_epoch)


def _polynomial_lr_init(self, optimizer, total_iters=5, power=1.0, last_epoch=-1):
    if total_iters <= 0:
        raise ValueError(f"total_iters must be positive, but got {total_iters}")
    _original_polynomial_lr_init(self, optimizer, total_iters, power, last_epoch)


def _cosine_lr_init(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
    if T_max <= 0:
        raise ValueError(f"T_max must be positive, but got {T_max}")
    _original_cosine_lr_init(self, optimizer, T_max, eta_min, last_epoch)


def _cyclic_lr_init(
    self,
    optimizer,
    base_lr,
    max_lr,
    step_size_up=2000.0,
    step_size_down=None,
    mode="triangular",
    gamma=1.0,
    scale_fn=None,
    scale_mode="cycle",
    cycle_momentum=True,
    base_momentum=0.1,
    max_momentum=0.9,
    last_epoch=-1,
):
    if step_size_up <= 0:
        raise ValueError(f"step_size_up must be positive, but got {step_size_up}")
    if step_size_down is not None and step_size_down <= 0:
        raise ValueError(
            f"step_size_down must be positive, but got {step_size_down}"
        )
    _original_cyclic_lr_init(
        self,
        optimizer,
        base_lr,
        max_lr,
        step_size_up,
        step_size_down,
        mode,
        gamma,
        scale_fn,
        scale_mode,
        cycle_momentum,
        base_momentum,
        max_momentum,
        last_epoch,
    )


def _one_cycle_lr_init(
    self,
    optimizer,
    max_lr,
    total_steps=None,
    epochs=None,
    steps_per_epoch=None,
    pct_start=0.3,
    anneal_strategy="cos",
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=25.0,
    final_div_factor=1e4,
    three_phase=False,
    last_epoch=-1,
):
    if div_factor <= 0:
        raise ValueError(f"div_factor must be positive, but got {div_factor}")
    if final_div_factor <= 0:
        raise ValueError(
            f"final_div_factor must be positive, but got {final_div_factor}"
        )
    _original_one_cycle_lr_init(
        self,
        optimizer,
        max_lr,
        total_steps,
        epochs,
        steps_per_epoch,
        pct_start,
        anneal_strategy,
        cycle_momentum,
        base_momentum,
        max_momentum,
        div_factor,
        final_div_factor,
        three_phase,
        last_epoch,
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
            lr_scheduler._update_param_group_val(param_group, "lr", new_lr)
        else:
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                lr_scheduler._update_param_group_val(param_group, "lr", new_lr)


def _composite_initial_step(self):
    self._step_count = 0
    self._last_lr = lr_scheduler._param_groups_val_list(self.optimizer, "lr")


def install() -> None:
    if getattr(lr_scheduler.LRScheduler, "_native_neo_fixes_installed", False):
        return
    with _install_lock:
        if getattr(lr_scheduler.LRScheduler, "_native_neo_fixes_installed", False):
            return
        lr_scheduler.LRScheduler.__init__ = _lrscheduler_init
        lr_scheduler.ConstantLR.__init__ = _constant_lr_init
        lr_scheduler.LinearLR.__init__ = _linear_lr_init
        lr_scheduler.StepLR.__init__ = _step_lr_init
        lr_scheduler.PolynomialLR.__init__ = _polynomial_lr_init
        lr_scheduler.CosineAnnealingLR.__init__ = _cosine_lr_init
        lr_scheduler.CyclicLR.__init__ = _cyclic_lr_init
        lr_scheduler.OneCycleLR.__init__ = _one_cycle_lr_init
        lr_scheduler.ReduceLROnPlateau._reduce_lr = _reduce_lr
        lr_scheduler.SequentialLR._initial_step = _composite_initial_step
        lr_scheduler.ChainedScheduler._initial_step = _composite_initial_step
        lr_scheduler.LRScheduler._native_neo_fixes_installed = True
