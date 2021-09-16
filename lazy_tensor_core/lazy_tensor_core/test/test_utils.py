from datetime import datetime
import multiprocessing
import time
import unittest

import lazy_tensor_core.core.lazy_model as ltm
import lazy_tensor_core.debug.metrics as met
import lazy_tensor_core.debug.metrics_compare_utils as mcu


def mp_test(func):
    """Wraps a `unittest.TestCase` function running it within an isolated process.

    Example::

      import lazy_tensor_core.test.test_utils as xtu
      import unittest

      class MyTest(unittest.TestCase):

        @xtu.mp_test
        def test_basic(self):
          ...

    Args:
      func (callable): The `unittest.TestCase` function to be wrapped.
    """

    def wrapper(*args, **kwargs):
        proc = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
        proc.start()
        proc.join()
        if isinstance(args[0], unittest.TestCase):
            args[0].assertEqual(proc.exitcode, 0)
        return proc.exitcode

    return wrapper


def _get_device_spec(device):
    ordinal = ltm.get_ordinal(defval=-1)
    return str(device) if ordinal < 0 else '{}/{}'.format(device, ordinal)


def write_to_summary(summary_writer,
                     global_step=None,
                     dict_to_write=None,
                     write_ltc_metrics=False):
    """Writes scalars to a Tensorboard SummaryWriter.

    Optionally writes lazy tensors perf metrics.

    Args:
      summary_writer (SummaryWriter): The Tensorboard SummaryWriter to write to.
        If None, no summary files will be written.
      global_step (int, optional): The global step value for these data points.
        If None, global_step will not be set for this datapoint.
      dict_to_write (dict, optional): Dict where key is the scalar name and value
        is the scalar value to be written to Tensorboard.
      write_ltc_metrics (bool, optional): If true, this method will retrieve
        performance metrics, parse them, and write them as scalars to Tensorboard.
    """
    if summary_writer is None:
        return
    if dict_to_write is None:
        dict_to_write = {}
    for k, v in dict_to_write.items():
        summary_writer.add_scalar(k, v, global_step)

    if write_ltc_metrics:
        metrics = mcu.parse_metrics_report(met.metrics_report())
        aten_ops_sum = 0
        for metric_name, metric_value in metrics.items():
            if metric_name.find('aten::') == 0:
                aten_ops_sum += metric_value
            summary_writer.add_scalar(metric_name, metric_value, global_step)
        summary_writer.add_scalar('aten_ops_sum', aten_ops_sum, global_step)


def close_summary_writer(summary_writer):
    """Flush and close a SummaryWriter.

    Args:
      summary_writer (SummaryWriter, optional): The Tensorboard SummaryWriter to
        close and flush. If None, no action is taken.
    """
    if summary_writer is not None:
        summary_writer.flush()
        summary_writer.close()


def get_summary_writer(logdir):
    """Initialize a Tensorboard SummaryWriter.

    Args:
      logdir (str): File location where logs will be written or None. If None, no
        writer is created.

    Returns:
      Instance of Tensorboard SummaryWriter.
    """
    if logdir:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=logdir)
        write_to_summary(
            writer, 0, dict_to_write={'TensorboardStartTimestamp': time.time()})
        return writer


def now(format='%H:%M:%S'):
    return datetime.now().strftime(format)


def print_training_update(device,
                          step,
                          loss,
                          rate,
                          global_rate,
                          epoch=None,
                          summary_writer=None):
    """Prints the training metrics at a given step.

    Args:
      device (torch.device): The device where these statistics came from.
      step_num (int): Current step number.
      loss (float): Current loss.
      rate (float): The examples/sec rate for the current batch.
      global_rate (float): The average examples/sec rate since training began.
      epoch (int, optional): The epoch number.
      summary_writer (SummaryWriter, optional): If provided, this method will
        write some of the provided statistics to Tensorboard.
    """
    update_data = [
        'Training', 'Device={}'.format(_get_device_spec(device)),
        'Epoch={}'.format(epoch) if epoch is not None else None,
        'Step={}'.format(step), 'Loss={:.5f}'.format(loss),
        'Rate={:.2f}'.format(rate), 'GlobalRate={:.2f}'.format(global_rate),
        'Time={}'.format(now())
    ]
    print('|', ' '.join(item for item in update_data if item), flush=True)
    if summary_writer:
        write_to_summary(
            summary_writer,
            dict_to_write={
                'examples/sec': rate,
                'average_examples/sec': global_rate,
            })


def print_test_update(device, accuracy, epoch=None, step=None):
    """Prints single-core test metrics.

    Args:
      device: Instance of `torch.device`.
      accuracy: Float.
    """
    update_data = [
        'Test', 'Device={}'.format(_get_device_spec(device)),
        'Step={}'.format(step) if step is not None else None,
        'Epoch={}'.format(epoch) if epoch is not None else None,
        'Accuracy={:.2f}'.format(accuracy) if accuracy is not None else None,
        'Time={}'.format(now())
    ]
    print('|', ' '.join(item for item in update_data if item), flush=True)
