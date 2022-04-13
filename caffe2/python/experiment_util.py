## @package experiment_util
# Module caffe2.python.experiment_util





import datetime
import time
import logging
import socket
import abc

from collections import OrderedDict
from future.utils import viewkeys, viewvalues

'''
Utilities for logging experiment run stats, such as accuracy
and loss over time for different runs. Runtime arguments are stored
in the log.

Optionally, ModelTrainerLog calls out to a logger to log to
an external log destination.
'''


class ExternalLogger(object):
    __metaclass__ = abc.ABCMeta

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
        self.external_loggers = None

        if external_loggers is not None:
            self.external_loggers = external_loggers
            if not isinstance(runtime_args, dict):
                runtime_args = dict(vars(runtime_args))
            runtime_args['experiment_id'] = self.experiment_id
            runtime_args['hostname'] = socket.gethostname()
            for logger in self.external_loggers:
                logger.set_runtime_args(runtime_args)
        else:
            self.external_loggers = []

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

        for k in sorted(viewkeys(additional_values)):
            logdict[k] = additional_values[k]

        # Write the headers if they are not written yet
        if self.headers is None:
            self.headers = list(viewkeys(logdict))
            self.logstr(",".join(self.headers))

        self.logstr(",".join(str(v) for v in viewvalues(logdict)))

        for logger in self.external_loggers:
            try:
                logger.log(logdict)
            except Exception as e:
                logging.warning(
                    "Failed to call ExternalLogger: {}".format(e), e)
