from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import time
import logging
import abc
import six

from collections import OrderedDict

'''
Utilities for logging experiment run stats, such as accuracy
and loss over time for different runs. Runtime arguments are stored
in the log.

Optionally, ModelTrainerLog calls out to an logger to log to
an external log destination.
'''


class ExternalLogger(object):
    six.add_metaclass(abc.ABCMeta)

    @abc.abstractmethod
    def set_runtime_args(self, runtime_args):
        """
            Set runtime arguments for the logger.
            runtime_args: dict of runtime arguments.
        """
        raise NotImplementedError(
            'Must define set_runtime_args function to use this base class'
        )

    @abc.abstractmethod
    def log(self, log_dict):
        """
            log a dict of key/values to an external destination
            log_dict: input dict
        """
        raise NotImplementedError(
            'Must define log function to use this base class'
        )


class ModelTrainerLog():

    def __init__(self, expname, runtime_args, external_loggers=None):
        now = datetime.datetime.fromtimestamp(time.time())
        self.experiment_id = \
            "{}_{}".format(expname, now.strftime('%Y%m%d_%H%M%S'))
        self.filename = "{}.log".format(self.experiment_id)
        self.logstr("# %s" % str(runtime_args))
        self.headers = None
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_input_count = 0

        if external_loggers is not None:
            self.external_loggers = external_loggers
            runtime_args = dict(vars(runtime_args))
            runtime_args['experiment_id'] = self.experiment_id
            for logger in self.external_loggers:
                logger.set_runtime_args(runtime_args)

    def logstr(self, str):
        with open(self.filename, "a") as f:
            f.write(str + "\n")
            f.close()
        logging.getLogger("experiment_logger").info(str)

    def log(self, input_count, batch_count, additional_values):
        logdict = OrderedDict()
        delta_t = time.time() - self.last_time
        delta_count = input_count - self.last_input_count
        self.last_time = time.time()
        self.last_input_count = input_count

        logdict['time_spent'] = delta_t
        logdict['cumulative_time_spent'] = time.time() - self.start_time
        logdict['input_count'] = delta_count
        logdict['cumulative_input_count'] = input_count
        logdict['cumulative_batch_count'] = batch_count
        if delta_t > 0:
            logdict['inputs_per_sec'] = delta_count / delta_t
        else:
            logdict['inputs_per_sec'] = 0.0

        for k in sorted(additional_values.keys()):
            logdict[k] = additional_values[k]

        # Write the headers if they are not written yet
        if self.headers is None:
            self.headers = logdict.keys()[:]
            self.logstr(",".join(self.headers))

        self.logstr(",".join([str(v) for v in logdict.values()]))

        if self.external_loggers:
            for logger in self.external_loggers:
                try:
                    logger.log(logdict)
                except Exception as e:
                    logging.warn(
                        "Failed to call ExternalLogger: {}".format(e), e)
