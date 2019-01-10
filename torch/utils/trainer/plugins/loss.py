from .monitor import Monitor


class LossMonitor(Monitor):
    stat_name = 'loss'

    def _get_value(self, iteration, input, target, output, loss):
        return loss[0]
