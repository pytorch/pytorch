from .plugin import Plugin


class Monitor(Plugin):

    def __init__(self, running_average=True, epoch_average=True, smoothing=0.7,
                 precision=None, number_format=None, unit=''):
        if precision is None:
            precision = 4
        if number_format is None:
            number_format = '.{}f'.format(precision)
        number_format = ':' + number_format
        super(Monitor, self).__init__([(1, 'iteration'), (1, 'epoch')])

        self.smoothing = smoothing
        self.with_running_average = running_average
        self.with_epoch_average = epoch_average

        self.log_format = number_format
        self.log_unit = unit
        self.log_epoch_fields = None
        self.log_iter_fields = ['{last' + number_format + '}' + unit]
        if self.with_running_average:
            self.log_iter_fields += [' ({running_avg' + number_format + '}' + unit + ')']
        if self.with_epoch_average:
            self.log_epoch_fields = ['{epoch_mean' + number_format + '}' + unit]

    def register(self, trainer):
        self.trainer = trainer
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['log_format'] = self.log_format
        stats['log_unit'] = self.log_unit
        stats['log_iter_fields'] = self.log_iter_fields
        if self.with_epoch_average:
            stats['log_epoch_fields'] = self.log_epoch_fields
        if self.with_epoch_average:
            stats['epoch_stats'] = (0, 0)

    def iteration(self, *args):
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['last'] = self._get_value(*args)

        if self.with_epoch_average:
            stats['epoch_stats'] = tuple(sum(t) for t in
                                         zip(stats['epoch_stats'], (stats['last'], 1)))

        if self.with_running_average:
            previous_avg = stats.get('running_avg', 0)
            stats['running_avg'] = previous_avg * self.smoothing + \
                stats['last'] * (1 - self.smoothing)

    def epoch(self, idx):
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        if self.with_epoch_average:
            epoch_stats = stats['epoch_stats']
            stats['epoch_mean'] = epoch_stats[0] / epoch_stats[1]
            stats['epoch_stats'] = (0, 0)
