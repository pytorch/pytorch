from collections import defaultdict
from .plugin import Plugin


class Logger(Plugin):
    alignment = 4
    separator = '#' * 80

    def __init__(self, fields, interval=None):
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        super(Logger, self).__init__(interval)
        self.field_widths = defaultdict(lambda: defaultdict(int))
        self.fields = list(map(lambda f: f.split('.'), fields))

    def _join_results(self, results):
        joined_out = map(lambda i: (i[0], ' '.join(i[1])), results)
        joined_fields = map(lambda i: '{}: {}'.format(i[0], i[1]), joined_out)
        return '\t'.join(joined_fields)

    def log(self, msg):
        print(msg)

    def register(self, trainer):
        self.trainer = trainer

    def gather_stats(self):
        result = {}
        return result

    def _align_output(self, field_idx, output):
        for output_idx, o in enumerate(output):
            if len(o) < self.field_widths[field_idx][output_idx]:
                num_spaces = self.field_widths[field_idx][output_idx] - len(o)
                output[output_idx] += ' ' * num_spaces
            else:
                self.field_widths[field_idx][output_idx] = len(o)

    def _gather_outputs(self, field, log_fields, stat_parent, stat, require_dict=False):
        output = []
        name = ''
        if isinstance(stat, dict):
            log_fields = stat.get(log_fields, [])
            name = stat.get('log_name', '.'.join(field))
            for f in log_fields:
                output.append(f.format(**stat))
        elif not require_dict:
            name = '.'.join(field)
            number_format = stat_parent.get('log_format', '')
            unit = stat_parent.get('log_unit', '')
            fmt = '{' + number_format + '}' + unit
            output.append(fmt.format(stat))
        return name, output

    def _log_all(self, log_fields, prefix=None, suffix=None, require_dict=False):
        results = []
        for field_idx, field in enumerate(self.fields):
            parent, stat = None, self.trainer.stats
            for f in field:
                parent, stat = stat, stat[f]
            name, output = self._gather_outputs(field, log_fields,
                                                parent, stat, require_dict)
            if not output:
                continue
            self._align_output(field_idx, output)
            results.append((name, output))
        if not results:
            return
        output = self._join_results(results)
        if prefix is not None:
            self.log(prefix)
        self.log(output)
        if suffix is not None:
            self.log(suffix)

    def iteration(self, *args):
        self._log_all('log_iter_fields')

    def epoch(self, epoch_idx):
        self._log_all('log_epoch_fields',
                      prefix=self.separator + '\nEpoch summary:',
                      suffix=self.separator,
                      require_dict=True)
