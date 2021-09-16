from __future__ import print_function

import os
import threading
import lazy_tensor_core.debug.metrics as met

_STEP_METRICS_FILE_LOCK = threading.Lock()

_TLS = threading.local()


def _counter():
    counter = getattr(_TLS, 'counter', 0)
    _TLS.counter = counter + 1
    return counter


def _extract_metrics_file():
    # Delay lazy_model import to avoid cross dependencies.
    import lazy_tensor_core.core.lazy_model as ltm
    metrics_file = os.environ.get('LTC_METRICS_FILE', None)
    if metrics_file is not None:
        ordinal = ltm.get_local_ordinal(defval=-1)
        if ordinal >= 0 and ltm.xrt_world_size() > 1:
            metrics_file = '{}.{}'.format(metrics_file, ordinal)
    return metrics_file


def _get_metrics_file():
    metrics_file = getattr(_TLS, 'metrics_file', '')
    if metrics_file == '':
        metrics_file = _extract_metrics_file()
        _TLS.metrics_file = metrics_file
    return metrics_file


def save_metrics(metrics_file=None):
    if metrics_file is None:
        metrics_file = _get_metrics_file()
    if metrics_file is not None:
        metrics_data = '[MetricsData; step={}]\n{}\n'.format(
            _counter(), met.metrics_report())
        if metrics_file == 'STDERR':
            print(metrics_data, file=sys.stderr)
        elif metrics_file == 'STDOUT':
            print(metrics_data)
        else:
            with _STEP_METRICS_FILE_LOCK:
                with open(metrics_file, 'a') as fd:
                    fd.write(metrics_data)
