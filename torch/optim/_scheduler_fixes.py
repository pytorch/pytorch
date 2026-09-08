from . import lr_scheduler


_original_constant_lr_init = lr_scheduler.ConstantLR.__init__
_original_linear_lr_init = lr_scheduler.LinearLR.__init__


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


def _composite_initial_step(self):
    self._step_count = 0
    self._last_lr = lr_scheduler._param_groups_val_list(self.optimizer, "lr")


def install() -> None:
    lr_scheduler.ConstantLR.__init__ = _constant_lr_init
    lr_scheduler.LinearLR.__init__ = _linear_lr_init
    lr_scheduler.SequentialLR._initial_step = _composite_initial_step
    lr_scheduler.ChainedScheduler._initial_step = _composite_initial_step
