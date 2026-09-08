from . import lr_scheduler


_original_linear_lr_init = lr_scheduler.LinearLR.__init__


def _linear_lr_init(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1):
    if total_iters <= 0:
        raise ValueError("total_iters must be greater than 0")
    _original_linear_lr_init(
        self, optimizer, start_factor, end_factor, total_iters, last_epoch
    )


def install() -> None:
    lr_scheduler.LinearLR.__init__ = _linear_lr_init
