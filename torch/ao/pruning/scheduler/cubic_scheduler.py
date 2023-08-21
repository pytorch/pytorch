import warnings

from .base_scheduler import BaseScheduler

__all__ = ["CubicSL"]

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


class CubicSL(BaseScheduler):
    r"""Sets the sparsity level of each parameter group to the final sl
    plus a given exponential function.

    .. math::

        s_i = s_f + (s_0 - s_f) \cdot \left( 1 - \frac{t - t_0}{n\Delta t} \right)^3

    where :math:`s_i` is the sparsity at epoch :math:`t`, :math;`s_f` is the final
    sparsity level, :math:`f(i)` is the function to be applied to the current epoch
    :math:`t`, initial epoch :math:`t_0`, and final epoch :math:`t_f`.
    :math:`\Delta t` is used to control how often the update of the sparsity level
    happens. By default,

    Args:
        sparsifier (BaseSparsifier): Wrapped sparsifier.
        init_sl (int, list): Initial level of sparsity
        init_t (int, list): Initial step, when pruning starts
        delta_t (int, list): Pruning frequency
        total_t (int, list): Total number of pruning steps
        initially_zero (bool, list): If True, sets the level of sparsity to 0
            before init_t (:math:`t_0`). Otherwise, the sparsity level before
            init_t (:math:`t_0`) is set to init_sl(:math:`s_0`)
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """
    def __init__(self,
                 sparsifier,
                 init_sl=0.0,
                 init_t=0,
                 delta_t=10,
                 total_t=100,
                 initially_zero=False,
                 last_epoch=-1,
                 verbose=False
                 ):
        self.sparsifier = sparsifier

        self.init_sl = self._make_sure_a_list(init_sl)
        self.init_t = self._make_sure_a_list(init_t)
        self.delta_t = self._make_sure_a_list(delta_t)
        self.total_t = self._make_sure_a_list(total_t)

        self.initially_zero = self._make_sure_a_list(initially_zero)

        super().__init__(sparsifier, last_epoch, verbose)

    @staticmethod
    def sparsity_compute_fn(s_0, s_f, t, t_0, dt, n, initially_zero=False):
        r""""Computes the current level of sparsity.

        Based on https://arxiv.org/pdf/1710.01878.pdf

        Args:
            s_0: Initial level of sparsity, :math:`s_i`
            s_f: Target level of sparsity, :math:`s_f`
            t: Current step, :math:`t`
            t_0: Initial step, :math:`t_0`
            dt: Pruning frequency, :math:`\Delta T`
            n: Pruning steps, :math:`n`
            initially_zero: Sets the level of sparsity to 0 before t_0.
                If False, sets to s_0

        Returns:
            The sparsity level :math:`s_t` at the current step :math:`t`
        """
        if initially_zero and t < t_0:
            return 0
        s_t = s_f + (s_0 - s_f) * (1.0 - (t - t_0) / (dt * n)) ** 3
        s_t = _clamp(s_t, s_0, s_f)
        return s_t

    def get_sl(self):
        if not self._get_sl_called_within_step:
            warnings.warn(
                "To get the last sparsity level computed by the scheduler, "
                "please use `get_last_sl()`.", stacklevel=TO_BE_DETERMINED)
        return [
            self.sparsity_compute_fn(
                s_0=initial_sparsity,
                s_f=final_sparsity,
                t=self.last_epoch,
                t_0=initial_epoch,
                dt=delta_epoch,
                n=interval_epochs,
                initially_zero=initially_zero
            ) for initial_sparsity, final_sparsity, initial_epoch, delta_epoch, interval_epochs, initially_zero in
            zip(
                self.init_sl,
                self.base_sl,
                self.init_t,
                self.delta_t,
                self.total_t,
                self.initially_zero
            )
        ]
