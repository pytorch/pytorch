from __future__ import absolute_import
import time

from .monitor import Monitor


class TimeMonitor(Monitor):
    stat_name = 'time'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('unit', 'ms')
        kwargs.setdefault('precision', 0)
        super(TimeMonitor, self).__init__(*args, **kwargs)
        self.last_time = None

    def _get_value(self, *args):
        if self.last_time:
            now = time.time()
            duration = now - self.last_time
            self.last_time = now
            return duration * 1000
        else:
            self.last_time = time.time()
            return 0
