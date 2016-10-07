from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import time

from collections import OrderedDict

'''
Utilities for logging experiment run stats, such as accuracy
and loss over time for different runs. Runtime arguments are stored
in the log.
'''


class ModelTrainerLog():

    def __init__(self, expname, runtime_args):
        now = datetime.datetime.fromtimestamp(time.time())
        self.experiment_id = now.strftime('%Y%m%d_%H%M%S')
        self.filename = "%s_%s.log" % (expname, self.experiment_id)
        self.logstr("# %s" % str(runtime_args))
        self.headers = None
        self.start_time = time.time()

    def logstr(self, str):
        with open(self.filename, "a") as f:
            f.write(str + "\n")
            f.close()
        print(str)

    def log(self, input_count, batch_count, additional_values):
        logdict = OrderedDict()
        logdict['time'] = time.time() - self.start_time
        logdict['input_counter'] = input_count
        logdict['batch_count'] = batch_count
        if logdict['time'] > 0:
            logdict['inputs_per_sec'] = input_count / logdict['time']
        else:
            logdict['inputs_per_sec'] = 0.0

        for k in sorted(additional_values.keys()):
            logdict[k] = additional_values[k]

        # Write the headers if they are not written yet
        if self.headers is None:
            self.headers = logdict.keys()[:]
            self.logstr(",".join(self.headers))

        self.logstr(",".join([str(v) for v in logdict.values()]))
