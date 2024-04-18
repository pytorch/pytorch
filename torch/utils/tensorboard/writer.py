"""Provide an API for writing protocol buffers to event files to be consumed by TensorBoard for visualization."""

import os
import time
from typing import List, Optional, Union, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto.event_pb2 import Event, SessionLog
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from ._convert_np import make_np
from ._embedding import get_embedding_info, make_mat, make_sprite, make_tsv, write_pbtxt
from ._onnx_graph import load_onnx_graph
from ._pytorch_graph import graph
from ._utils import figure_to_image
from .summary import (
    audio,
    custom_scalars,
    histogram,
    histogram_raw,
    hparams,
    image,
    image_boxes,
    mesh,
    pr_curve,
    pr_curve_raw,
    scalar,
    tensor_proto,
    text,
    video,
)

__all__ = ["FileWriter", "SummaryWriter"]


class FileWriter:
    """Writes protocol buffers to event files to be consumed by TensorBoard.

    The `FileWriter` class provides a mechanism to create an event file in a
    given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """

    def __init__(self, log_dir, max_queue=10, flush_secs=120, filename_suffix=""):
        """Create a `FileWriter` and an event file.

        On construction the writer creates a new event file in `log_dir`.
        The other arguments to the constructor control the asynchronous writes to
        the event file.

        Args:
          log_dir: A string. Directory where event file will be written.
          max_queue: Integer. Size of the queue for pending events and
            summaries before one of the 'add' calls forces a flush to disk.
            Default is ten items.
          flush_secs: Number. How often, in seconds, to flush the
            pending events and summaries to disk. Default is every two minutes.
          filename_suffix: A string. Suffix added to all event filenames
            in the log_dir directory. More details on filename construction in
            tensorboard.summary.writer.event_file_writer.EventFileWriter.
        """
        # Sometimes PosixPath is passed in and we need to coerce it to
        # a string in all cases
        # TODO: See if we can remove this in the future if we are
        # actually the ones passing in a PosixPath
        log_dir = str(log_dir)
        self.event_writer = EventFileWriter(
            log_dir, max_queue, flush_secs, filename_suffix
        )

    def get_logdir(self):
        """Return the directory where event file will be written."""
        return self.event_writer.get_logdir()

    def add_event(self, event, step=None, walltime=None):
        """Add an event to the event file.

        Args:
          event: An `Event` protocol buffer.
          step: Number. Optional global step value for training process
            to record with the event.
          walltime: float. Optional walltime to override the default (current)
            walltime (from time.time()) seconds after epoch
        """
        event.wall_time = time.time() if walltime is None else walltime
        if step is not None:
            # Make sure step is converted from numpy or other formats
            # since protobuf might not convert depending on version
            event.step = int(step)
        self.event_writer.add_event(event)

    def add_summary(self, summary, global_step=None, walltime=None):
        """Add a `Summary` protocol buffer to the event file.

        This method wraps the provided summary in an `Event` protocol buffer
        and adds it to the event file.

        Args:
          summary: A `Summary` protocol buffer.
          global_step: Number. Optional global step value for training process
            to record with the summary.
          walltime: float. Optional walltime to override the default (current)
            walltime (from time.time()) seconds after epoch
        """
        event = event_pb2.Event(summary=summary)
        self.add_event(event, global_step, walltime)

    def add_graph(self, graph_profile, walltime=None):
        """Add a `Graph` and step stats protocol buffer to the event file.

        Args:
          graph_profile: A `Graph` and step stats protocol buffer.
          walltime: float. Optional walltime to override the default (current)
            walltime (from time.time()) seconds after epoch
        """
        graph = graph_profile[0]
        stepstats = graph_profile[1]
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self.add_event(event, None, walltime)

        trm = event_pb2.TaggedRunMetadata(
            tag="step1", run_metadata=stepstats.SerializeToString()
        )
        event = event_pb2.Event(tagged_run_metadata=trm)
        self.add_event(event, None, walltime)

    def add_onnx_graph(self, graph, walltime=None):
        """Add a `Graph` protocol buffer to the event file.

        Args:
          graph: A `Graph` protocol buffer.
          walltime: float. Optional walltime to override the default (current)
            _get_file_writerfrom time.time())
        """
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self.add_event(event, None, walltime)

    def flush(self):
        """Flushes the event file to disk.

        Call this method to make sure that all pending events have been written to
        disk.
        """
        self.event_writer.flush()

    def close(self):
        """Flushes the event file to disk and close the file.

        Call this method when you do not need the summary writer anymore.
        """
        self.event_writer.close()

    def reopen(self):
        """Reopens the EventFileWriter.

        Can be called after `close()` to add more events in the same directory.
        The events will go into a new events file.
        Does nothing if the EventFileWriter was not closed.
        """
        self.event_writer.reopen()


class SummaryWriter:
    """Writes entries directly to event files in the log_dir to be consumed by TensorBoard.

    The `SummaryWriter` class provides a high-level API to create an event file
    in a given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """

    def __init__(
        self,
        log_dir=None,
        comment="",
        purge_step=None,
        max_queue=10,
        flush_secs=120,
        filename_suffix="",
    ):
        """Create a `SummaryWriter` that will write out events and summaries to the event file.

        Args:
            log_dir (str): Save directory location. Default is
              runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each run.
              Use hierarchical folder structure to compare
              between runs easily. e.g. pass in 'runs/exp1', 'runs/exp2', etc.
              for each new experiment to compare across them.
            comment (str): Comment log_dir suffix appended to the default
              ``log_dir``. If ``log_dir`` is assigned, this argument has no effect.
            purge_step (int):
              When logging crashes at step :math:`T+X` and restarts at step :math:`T`,
              any events whose global_step larger or equal to :math:`T` will be
              purged and hidden from TensorBoard.
              Note that crashed and resumed experiments should have the same ``log_dir``.
            max_queue (int): Size of the queue for pending events and
              summaries before one of the 'add' calls forces a flush to disk.
              Default is ten items.
            flush_secs (int): How often, in seconds, to flush the
              pending events and summaries to disk. Default is every two minutes.
            filename_suffix (str): Suffix added to all event filenames in
              the log_dir directory. More details on filename construction in
              tensorboard.summary.writer.event_file_writer.EventFileWriter.

        Examples::

            from torch.utils.tensorboard import SummaryWriter

            # create a summary writer with automatically generated folder name.
            writer = SummaryWriter()
            # folder location: runs/May04_22-14-54_s-MacBook-Pro.local/

            # create a summary writer using the specified folder name.
            writer = SummaryWriter("my_experiment")
            # folder location: my_experiment

            # create a summary writer with comment appended.
            writer = SummaryWriter(comment="LR_0.1_BATCH_16")
            # folder location: runs/May04_22-14-54_s-MacBook-Pro.localLR_0.1_BATCH_16/

        """
        torch._C._log_api_usage_once("tensorboard.create.summarywriter")
        if not log_dir:
            import socket
            from datetime import datetime

            current_time = datetime.now().strftime("%b%d_%H-%M-%S")
            log_dir = os.path.join(
                "runs", current_time + "_" + socket.gethostname() + comment
            )
        self.log_dir = log_dir
        self.purge_step = purge_step
        self.max_queue = max_queue
        self.flush_secs = flush_secs
        self.filename_suffix = filename_suffix

        # Initialize the file writers, but they can be cleared out on close
        # and recreated later as needed.
        self.file_writer = self.all_writers = None
        self._get_file_writer()

        # Create default bins for histograms, see generate_testdata.py in tensorflow/tensorboard
        v = 1e-12
        buckets = []
        neg_buckets = []
        while v < 1e20:
            buckets.append(v)
            neg_buckets.append(-v)
            v *= 1.1
        self.default_bins = neg_buckets[::-1] + [0] + buckets

    def _check_caffe2_blob(self, item):
        """
        Check if the input is a string representing a Caffe2 blob name.

        Caffe2 users have the option of passing a string representing the name of a blob
        in the workspace instead of passing the actual Tensor/array containing the numeric values.
        Thus, we need to check if we received a string as input
        instead of an actual Tensor/array, and if so, we need to fetch the Blob
        from the workspace corresponding to that name. Fetching can be done with the
        following:

        from caffe2.python import workspace (if not already imported)
        workspace.FetchBlob(blob_name)
        workspace.FetchBlobs([blob_name1, blob_name2, ...])
        """
        return isinstance(item, str)

    def _get_file_writer(self):
        """Return the default FileWriter instance. Recreates it if closed."""
        if self.all_writers is None or self.file_writer is None:
            self.file_writer = FileWriter(
                self.log_dir, self.max_queue, self.flush_secs, self.filename_suffix
            )
            self.all_writers = {self.file_writer.get_logdir(): self.file_writer}
            if self.purge_step is not None:
                most_recent_step = self.purge_step
                self.file_writer.add_event(
                    Event(step=most_recent_step, file_version="brain.Event:2")
                )
                self.file_writer.add_event(
                    Event(
                        step=most_recent_step,
                        session_log=SessionLog(status=SessionLog.START),
                    )
                )
                self.purge_step = None
        return self.file_writer

    def get_logdir(self):
        """Return the directory where event files will be written."""
        return self.log_dir

    def add_hparams(
        self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None, global_step=None
    ):
        """Add a set of hyperparameters to be compared in TensorBoard.

        Args:
            hparam_dict (dict): Each key-value pair in the dictionary is the
              name of the hyper parameter and it's corresponding value.
              The type of the value can be one of `bool`, `string`, `float`,
              `int`, or `None`.
            metric_dict (dict): Each key-value pair in the dictionary is the
              name of the metric and it's corresponding value. Note that the key used
              here should be unique in the tensorboard record. Otherwise the value
              you added by ``add_scalar`` will be displayed in hparam plugin. In most
              cases, this is unwanted.
            hparam_domain_discrete: (Optional[Dict[str, List[Any]]]) A dictionary that
              contains names of the hyperparameters and all discrete values they can hold
            run_name (str): Name of the run, to be included as part of the logdir.
              If unspecified, will use current timestamp.
            global_step (int): Global step value to record

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            with SummaryWriter() as w:
                for i in range(5):
                    w.add_hparams({'lr': 0.1*i, 'bsize': i},
                                  {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})

        Expected result:

        .. image:: _static/img/tensorboard/add_hparam.png
           :scale: 50 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        if not run_name:
            run_name = str(time.time())
        logdir = os.path.join(self._get_file_writer().get_logdir(), run_name)
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp, global_step)
            w_hp.file_writer.add_summary(ssi, global_step)
            w_hp.file_writer.add_summary(sei, global_step)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v, global_step)

    def add_scalar(
        self,
        tag,
        scalar_value,
        global_step=None,
        walltime=None,
        new_style=False,
        double_precision=False,
    ):
        """Add scalar data to summary.

        Args:
            tag (str): Data identifier
            scalar_value (float or string/blobname): Value to save
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              with seconds after epoch of event
            new_style (boolean): Whether to use new style (tensor field) or old
              style (simple_value field). New style could lead to faster data loading.
        Examples::

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            x = range(100)
            for i in x:
                writer.add_scalar('y=2x', i * 2, i)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_scalar.png
           :scale: 50 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_scalar")
        if self._check_caffe2_blob(scalar_value):
            from caffe2.python import workspace

            scalar_value = workspace.FetchBlob(scalar_value)

        summary = scalar(
            tag, scalar_value, new_style=new_style, double_precision=double_precision
        )
        self._get_file_writer().add_summary(summary, global_step, walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        """Add many scalar data to summary.

        Args:
            main_tag (str): The parent name for the tags
            tag_scalar_dict (dict): Key-value pair storing the tag and corresponding values
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            r = 5
            for i in range(100):
                writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                                'xcosx':i*np.cos(i/r),
                                                'tanx': np.tan(i/r)}, i)
            writer.close()
            # This call adds three values to the same scalar plot with the tag
            # 'run_14h' in TensorBoard's scalar section.

        Expected result:

        .. image:: _static/img/tensorboard/add_scalars.png
           :scale: 50 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_scalars")
        walltime = time.time() if walltime is None else walltime
        fw_logdir = self._get_file_writer().get_logdir()
        for tag, scalar_value in tag_scalar_dict.items():
            fw_tag = fw_logdir + "/" + main_tag.replace("/", "_") + "_" + tag
            assert self.all_writers is not None
            if fw_tag in self.all_writers.keys():
                fw = self.all_writers[fw_tag]
            else:
                fw = FileWriter(
                    fw_tag, self.max_queue, self.flush_secs, self.filename_suffix
                )
                self.all_writers[fw_tag] = fw
            if self._check_caffe2_blob(scalar_value):
                from caffe2.python import workspace

                scalar_value = workspace.FetchBlob(scalar_value)
            fw.add_summary(scalar(main_tag, scalar_value), global_step, walltime)

    def add_tensor(
        self,
        tag,
        tensor,
        global_step=None,
        walltime=None,
    ):
        """Add tensor data to summary.

        Args:
            tag (str): Data identifier
            tensor (torch.Tensor): tensor to save
            global_step (int): Global step value to record
        Examples::

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            x = torch.tensor([1,2,3])
            writer.add_scalar('x', x)
            writer.close()

        Expected result:
            Summary::tensor::float_val [1,2,3]
                   ::tensor::shape [3]
                   ::tag 'x'

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_tensor")
        if self._check_caffe2_blob(tensor):
            from caffe2.python import workspace

            tensor = torch.tensor(workspace.FetchBlob(tensor))

        summary = tensor_proto(tag, tensor)
        self._get_file_writer().add_summary(summary, global_step, walltime)

    def add_histogram(
        self,
        tag,
        values,
        global_step=None,
        bins="tensorflow",
        walltime=None,
        max_bins=None,
    ):
        """Add histogram to summary.

        Args:
            tag (str): Data identifier
            values (torch.Tensor, numpy.ndarray, or string/blobname): Values to build histogram
            global_step (int): Global step value to record
            bins (str): One of {'tensorflow','auto', 'fd', ...}. This determines how the bins are made. You can find
              other options in: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            import numpy as np
            writer = SummaryWriter()
            for i in range(10):
                x = np.random.random(1000)
                writer.add_histogram('distribution centers', x + i, i)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_histogram.png
           :scale: 50 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_histogram")
        if self._check_caffe2_blob(values):
            from caffe2.python import workspace

            values = workspace.FetchBlob(values)
        if isinstance(bins, str) and bins == "tensorflow":
            bins = self.default_bins
        self._get_file_writer().add_summary(
            histogram(tag, values, bins, max_bins=max_bins), global_step, walltime
        )

    def add_histogram_raw(
        self,
        tag,
        min,
        max,
        num,
        sum,
        sum_squares,
        bucket_limits,
        bucket_counts,
        global_step=None,
        walltime=None,
    ):
        """Add histogram with raw data.

        Args:
            tag (str): Data identifier
            min (float or int): Min value
            max (float or int): Max value
            num (int): Number of values
            sum (float or int): Sum of all values
            sum_squares (float or int): Sum of squares for all values
            bucket_limits (torch.Tensor, numpy.ndarray): Upper value per bucket.
              The number of elements of it should be the same as `bucket_counts`.
            bucket_counts (torch.Tensor, numpy.ndarray): Number of values per bucket
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
            see: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/histogram/README.md

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            import numpy as np
            writer = SummaryWriter()
            dummy_data = []
            for idx, value in enumerate(range(50)):
                dummy_data += [idx + 0.001] * value

            bins = list(range(50+2))
            bins = np.array(bins)
            values = np.array(dummy_data).astype(float).reshape(-1)
            counts, limits = np.histogram(values, bins=bins)
            sum_sq = values.dot(values)
            writer.add_histogram_raw(
                tag='histogram_with_raw_data',
                min=values.min(),
                max=values.max(),
                num=len(values),
                sum=values.sum(),
                sum_squares=sum_sq,
                bucket_limits=limits[1:].tolist(),
                bucket_counts=counts.tolist(),
                global_step=0)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_histogram_raw.png
           :scale: 50 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_histogram_raw")
        if len(bucket_limits) != len(bucket_counts):
            raise ValueError(
                "len(bucket_limits) != len(bucket_counts), see the document."
            )
        self._get_file_writer().add_summary(
            histogram_raw(
                tag, min, max, num, sum, sum_squares, bucket_limits, bucket_counts
            ),
            global_step,
            walltime,
        )

    def add_image(
        self, tag, img_tensor, global_step=None, walltime=None, dataformats="CHW"
    ):
        """Add image data to summary.

        Note that this requires the ``pillow`` package.

        Args:
            tag (str): Data identifier
            img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
            dataformats (str): Image data format specification of the form
              CHW, HWC, HW, WH, etc.
        Shape:
            img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
            convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
            Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
            corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            import numpy as np
            img = np.zeros((3, 100, 100))
            img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
            img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

            img_HWC = np.zeros((100, 100, 3))
            img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
            img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

            writer = SummaryWriter()
            writer.add_image('my_image', img, 0)

            # If you have non-default dimension setting, set the dataformats argument.
            writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_image.png
           :scale: 50 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_image")
        if self._check_caffe2_blob(img_tensor):
            from caffe2.python import workspace

            img_tensor = workspace.FetchBlob(img_tensor)
        self._get_file_writer().add_summary(
            image(tag, img_tensor, dataformats=dataformats), global_step, walltime
        )

    def add_images(
        self, tag, img_tensor, global_step=None, walltime=None, dataformats="NCHW"
    ):
        """Add batched image data to summary.

        Note that this requires the ``pillow`` package.

        Args:
            tag (str): Data identifier
            img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
            dataformats (str): Image data format specification of the form
              NCHW, NHWC, CHW, HWC, HW, WH, etc.
        Shape:
            img_tensor: Default is :math:`(N, 3, H, W)`. If ``dataformats`` is specified, other shape will be
            accepted. e.g. NCHW or NHWC.

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            import numpy as np

            img_batch = np.zeros((16, 3, 100, 100))
            for i in range(16):
                img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
                img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i

            writer = SummaryWriter()
            writer.add_images('my_image_batch', img_batch, 0)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_images.png
           :scale: 30 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_images")
        if self._check_caffe2_blob(img_tensor):
            from caffe2.python import workspace

            img_tensor = workspace.FetchBlob(img_tensor)
        self._get_file_writer().add_summary(
            image(tag, img_tensor, dataformats=dataformats), global_step, walltime
        )

    def add_image_with_boxes(
        self,
        tag,
        img_tensor,
        box_tensor,
        global_step=None,
        walltime=None,
        rescale=1,
        dataformats="CHW",
        labels=None,
    ):
        """Add image and draw bounding boxes on the image.

        Args:
            tag (str): Data identifier
            img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
            box_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Box data (for detected objects)
              box should be represented as [x1, y1, x2, y2].
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
            rescale (float): Optional scale override
            dataformats (str): Image data format specification of the form
              NCHW, NHWC, CHW, HWC, HW, WH, etc.
            labels (list of string): The label to be shown for each bounding box.
        Shape:
            img_tensor: Default is :math:`(3, H, W)`. It can be specified with ``dataformats`` argument.
            e.g. CHW or HWC

            box_tensor: (torch.Tensor, numpy.ndarray, or string/blobname): NX4,  where N is the number of
            boxes and each 4 elements in a row represents (xmin, ymin, xmax, ymax).
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_image_with_boxes")
        if self._check_caffe2_blob(img_tensor):
            from caffe2.python import workspace

            img_tensor = workspace.FetchBlob(img_tensor)
        if self._check_caffe2_blob(box_tensor):
            from caffe2.python import workspace

            box_tensor = workspace.FetchBlob(box_tensor)
        if labels is not None:
            if isinstance(labels, str):
                labels = [labels]
            if len(labels) != box_tensor.shape[0]:
                labels = None
        self._get_file_writer().add_summary(
            image_boxes(
                tag,
                img_tensor,
                box_tensor,
                rescale=rescale,
                dataformats=dataformats,
                labels=labels,
            ),
            global_step,
            walltime,
        )

    def add_figure(
        self,
        tag: str,
        figure: Union["Figure", List["Figure"]],
        global_step: Optional[int] = None,
        close: bool = True,
        walltime: Optional[float] = None
    ) -> None:
        """Render matplotlib figure into an image and add it to summary.

        Note that this requires the ``matplotlib`` package.

        Args:
            tag: Data identifier
            figure: Figure or a list of figures
            global_step: Global step value to record
            close: Flag to automatically close the figure
            walltime: Optional override default walltime (time.time())
              seconds after epoch of event
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_figure")
        if isinstance(figure, list):
            self.add_image(
                tag,
                figure_to_image(figure, close),
                global_step,
                walltime,
                dataformats="NCHW",
            )
        else:
            self.add_image(
                tag,
                figure_to_image(figure, close),
                global_step,
                walltime,
                dataformats="CHW",
            )

    def add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None):
        """Add video data to summary.

        Note that this requires the ``moviepy`` package.

        Args:
            tag (str): Data identifier
            vid_tensor (torch.Tensor): Video data
            global_step (int): Global step value to record
            fps (float or int): Frames per second
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
        Shape:
            vid_tensor: :math:`(N, T, C, H, W)`. The values should lie in [0, 255] for type `uint8` or [0, 1] for type `float`.
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_video")
        self._get_file_writer().add_summary(
            video(tag, vid_tensor, fps), global_step, walltime
        )

    def add_audio(
        self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None
    ):
        """Add audio data to summary.

        Args:
            tag (str): Data identifier
            snd_tensor (torch.Tensor): Sound data
            global_step (int): Global step value to record
            sample_rate (int): sample rate in Hz
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
        Shape:
            snd_tensor: :math:`(1, L)`. The values should lie between [-1, 1].
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_audio")
        if self._check_caffe2_blob(snd_tensor):
            from caffe2.python import workspace

            snd_tensor = workspace.FetchBlob(snd_tensor)
        self._get_file_writer().add_summary(
            audio(tag, snd_tensor, sample_rate=sample_rate), global_step, walltime
        )

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        """Add text data to summary.

        Args:
            tag (str): Data identifier
            text_string (str): String to save
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
        Examples::

            writer.add_text('lstm', 'This is an lstm', 0)
            writer.add_text('rnn', 'This is an rnn', 10)
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_text")
        self._get_file_writer().add_summary(
            text(tag, text_string), global_step, walltime
        )

    def add_onnx_graph(self, prototxt):
        torch._C._log_api_usage_once("tensorboard.logging.add_onnx_graph")
        self._get_file_writer().add_onnx_graph(load_onnx_graph(prototxt))

    def add_graph(
        self, model, input_to_model=None, verbose=False, use_strict_trace=True
    ):
        """Add graph data to summary.

        Args:
            model (torch.nn.Module): Model to draw.
            input_to_model (torch.Tensor or list of torch.Tensor): A variable or a tuple of
                variables to be fed.
            verbose (bool): Whether to print graph structure in console.
            use_strict_trace (bool): Whether to pass keyword argument `strict` to
                `torch.jit.trace`. Pass False when you want the tracer to
                record your mutable container types (list, dict)
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_graph")
        if hasattr(model, "forward"):
            # A valid PyTorch model should have a 'forward' method
            self._get_file_writer().add_graph(
                graph(model, input_to_model, verbose, use_strict_trace)
            )
        else:
            # Caffe2 models do not have the 'forward' method
            from caffe2.proto import caffe2_pb2
            from caffe2.python import core

            from ._caffe2_graph import (
                model_to_graph_def,
                nets_to_graph_def,
                protos_to_graph_def,
            )

            if isinstance(model, list):
                if isinstance(model[0], core.Net):
                    current_graph = nets_to_graph_def(model)
                elif isinstance(model[0], caffe2_pb2.NetDef):
                    current_graph = protos_to_graph_def(model)
            else:
                # Handles cnn.CNNModelHelper, model_helper.ModelHelper
                current_graph = model_to_graph_def(model)
            event = event_pb2.Event(graph_def=current_graph.SerializeToString())  # type: ignore[possibly-undefined]
            self._get_file_writer().add_event(event)

    @staticmethod
    def _encode(rawstr):
        # I'd use urllib but, I'm unsure about the differences from python3 to python2, etc.
        retval = rawstr
        retval = retval.replace("%", f"%{ord('%'):02x}")
        retval = retval.replace("/", f"%{ord('/'):02x}")
        retval = retval.replace("\\", "%%%02x" % (ord("\\")))
        return retval

    def add_embedding(
        self,
        mat,
        metadata=None,
        label_img=None,
        global_step=None,
        tag="default",
        metadata_header=None,
    ):
        """Add embedding projector data to summary.

        Args:
            mat (torch.Tensor or numpy.ndarray): A matrix which each row is the feature vector of the data point
            metadata (list): A list of labels, each element will be convert to string
            label_img (torch.Tensor): Images correspond to each data point
            global_step (int): Global step value to record
            tag (str): Name for the embedding
        Shape:
            mat: :math:`(N, D)`, where N is number of data and D is feature dimension

            label_img: :math:`(N, C, H, W)`

        Examples::

            import keyword
            import torch
            meta = []
            while len(meta)<100:
                meta = meta+keyword.kwlist # get some strings
            meta = meta[:100]

            for i, v in enumerate(meta):
                meta[i] = v+str(i)

            label_img = torch.rand(100, 3, 10, 32)
            for i in range(100):
                label_img[i]*=i/100.0

            writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
            writer.add_embedding(torch.randn(100, 5), label_img=label_img)
            writer.add_embedding(torch.randn(100, 5), metadata=meta)
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_embedding")
        mat = make_np(mat)
        if global_step is None:
            global_step = 0
            # clear pbtxt?

        # Maybe we should encode the tag so slashes don't trip us up?
        # I don't think this will mess us up, but better safe than sorry.
        subdir = f"{str(global_step).zfill(5)}/{self._encode(tag)}"
        save_path = os.path.join(self._get_file_writer().get_logdir(), subdir)

        fs = tf.io.gfile
        if fs.exists(save_path):
            if fs.isdir(save_path):
                print(
                    "warning: Embedding dir exists, did you set global_step for add_embedding()?"
                )
            else:
                raise Exception(
                    f"Path: `{save_path}` exists, but is a file. Cannot proceed."
                )
        else:
            fs.makedirs(save_path)

        if metadata is not None:
            assert mat.shape[0] == len(
                metadata
            ), "#labels should equal with #data points"
            make_tsv(metadata, save_path, metadata_header=metadata_header)

        if label_img is not None:
            assert (
                mat.shape[0] == label_img.shape[0]
            ), "#images should equal with #data points"
            make_sprite(label_img, save_path)

        assert (
            mat.ndim == 2
        ), "mat should be 2D, where mat.size(0) is the number of data points"
        make_mat(mat, save_path)

        # Filesystem doesn't necessarily have append semantics, so we store an
        # internal buffer to append to and re-write whole file after each
        # embedding is added
        if not hasattr(self, "_projector_config"):
            self._projector_config = ProjectorConfig()
        embedding_info = get_embedding_info(
            metadata, label_img, subdir, global_step, tag
        )
        self._projector_config.embeddings.extend([embedding_info])

        from google.protobuf import text_format

        config_pbtxt = text_format.MessageToString(self._projector_config)
        write_pbtxt(self._get_file_writer().get_logdir(), config_pbtxt)

    def add_pr_curve(
        self,
        tag,
        labels,
        predictions,
        global_step=None,
        num_thresholds=127,
        weights=None,
        walltime=None,
    ):
        """Add precision recall curve.

        Plotting a precision-recall curve lets you understand your model's
        performance under different threshold settings. With this function,
        you provide the ground truth labeling (T/F) and prediction confidence
        (usually the output of your model) for each target. The TensorBoard UI
        will let you choose the threshold interactively.

        Args:
            tag (str): Data identifier
            labels (torch.Tensor, numpy.ndarray, or string/blobname):
              Ground truth data. Binary label for each element.
            predictions (torch.Tensor, numpy.ndarray, or string/blobname):
              The probability that an element be classified as true.
              Value should be in [0, 1]
            global_step (int): Global step value to record
            num_thresholds (int): Number of thresholds used to draw the curve.
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            import numpy as np
            labels = np.random.randint(2, size=100)  # binary label
            predictions = np.random.rand(100)
            writer = SummaryWriter()
            writer.add_pr_curve('pr_curve', labels, predictions, 0)
            writer.close()

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_pr_curve")
        labels, predictions = make_np(labels), make_np(predictions)
        self._get_file_writer().add_summary(
            pr_curve(tag, labels, predictions, num_thresholds, weights),
            global_step,
            walltime,
        )

    def add_pr_curve_raw(
        self,
        tag,
        true_positive_counts,
        false_positive_counts,
        true_negative_counts,
        false_negative_counts,
        precision,
        recall,
        global_step=None,
        num_thresholds=127,
        weights=None,
        walltime=None,
    ):
        """Add precision recall curve with raw data.

        Args:
            tag (str): Data identifier
            true_positive_counts (torch.Tensor, numpy.ndarray, or string/blobname): true positive counts
            false_positive_counts (torch.Tensor, numpy.ndarray, or string/blobname): false positive counts
            true_negative_counts (torch.Tensor, numpy.ndarray, or string/blobname): true negative counts
            false_negative_counts (torch.Tensor, numpy.ndarray, or string/blobname): false negative counts
            precision (torch.Tensor, numpy.ndarray, or string/blobname): precision
            recall (torch.Tensor, numpy.ndarray, or string/blobname): recall
            global_step (int): Global step value to record
            num_thresholds (int): Number of thresholds used to draw the curve.
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
            see: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/README.md
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_pr_curve_raw")
        self._get_file_writer().add_summary(
            pr_curve_raw(
                tag,
                true_positive_counts,
                false_positive_counts,
                true_negative_counts,
                false_negative_counts,
                precision,
                recall,
                num_thresholds,
                weights,
            ),
            global_step,
            walltime,
        )

    def add_custom_scalars_multilinechart(
        self, tags, category="default", title="untitled"
    ):
        """Shorthand for creating multilinechart. Similar to ``add_custom_scalars()``, but the only necessary argument is *tags*.

        Args:
            tags (list): list of tags that have been used in ``add_scalar()``

        Examples::

            writer.add_custom_scalars_multilinechart(['twse/0050', 'twse/2330'])
        """
        torch._C._log_api_usage_once(
            "tensorboard.logging.add_custom_scalars_multilinechart"
        )
        layout = {category: {title: ["Multiline", tags]}}
        self._get_file_writer().add_summary(custom_scalars(layout))

    def add_custom_scalars_marginchart(
        self, tags, category="default", title="untitled"
    ):
        """Shorthand for creating marginchart.

        Similar to ``add_custom_scalars()``, but the only necessary argument is *tags*,
        which should have exactly 3 elements.

        Args:
            tags (list): list of tags that have been used in ``add_scalar()``

        Examples::

            writer.add_custom_scalars_marginchart(['twse/0050', 'twse/2330', 'twse/2006'])
        """
        torch._C._log_api_usage_once(
            "tensorboard.logging.add_custom_scalars_marginchart"
        )
        assert len(tags) == 3
        layout = {category: {title: ["Margin", tags]}}
        self._get_file_writer().add_summary(custom_scalars(layout))

    def add_custom_scalars(self, layout):
        """Create special chart by collecting charts tags in 'scalars'.

        NOTE: This function can only be called once for each SummaryWriter() object.

        Because it only provides metadata to tensorboard, the function can be called before or after the training loop.

        Args:
            layout (dict): {categoryName: *charts*}, where *charts* is also a dictionary
              {chartName: *ListOfProperties*}. The first element in *ListOfProperties* is the chart's type
              (one of **Multiline** or **Margin**) and the second element should be a list containing the tags
              you have used in add_scalar function, which will be collected into the new chart.

        Examples::

            layout = {'Taiwan':{'twse':['Multiline',['twse/0050', 'twse/2330']]},
                         'USA':{ 'dow':['Margin',   ['dow/aaa', 'dow/bbb', 'dow/ccc']],
                              'nasdaq':['Margin',   ['nasdaq/aaa', 'nasdaq/bbb', 'nasdaq/ccc']]}}

            writer.add_custom_scalars(layout)
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_custom_scalars")
        self._get_file_writer().add_summary(custom_scalars(layout))

    def add_mesh(
        self,
        tag,
        vertices,
        colors=None,
        faces=None,
        config_dict=None,
        global_step=None,
        walltime=None,
    ):
        """Add meshes or 3D point clouds to TensorBoard.

        The visualization is based on Three.js,
        so it allows users to interact with the rendered object. Besides the basic definitions
        such as vertices, faces, users can further provide camera parameter, lighting condition, etc.
        Please check https://threejs.org/docs/index.html#manual/en/introduction/Creating-a-scene for
        advanced usage.

        Args:
            tag (str): Data identifier
            vertices (torch.Tensor): List of the 3D coordinates of vertices.
            colors (torch.Tensor): Colors for each vertex
            faces (torch.Tensor): Indices of vertices within each triangle. (Optional)
            config_dict: Dictionary with ThreeJS classes names and configuration.
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event

        Shape:
            vertices: :math:`(B, N, 3)`. (batch, number_of_vertices, channels)

            colors: :math:`(B, N, 3)`. The values should lie in [0, 255] for type `uint8` or [0, 1] for type `float`.

            faces: :math:`(B, N, 3)`. The values should lie in [0, number_of_vertices] for type `uint8`.

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            vertices_tensor = torch.as_tensor([
                [1, 1, 1],
                [-1, -1, 1],
                [1, -1, -1],
                [-1, 1, -1],
            ], dtype=torch.float).unsqueeze(0)
            colors_tensor = torch.as_tensor([
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 255],
            ], dtype=torch.int).unsqueeze(0)
            faces_tensor = torch.as_tensor([
                [0, 2, 3],
                [0, 3, 1],
                [0, 1, 2],
                [1, 3, 2],
            ], dtype=torch.int).unsqueeze(0)

            writer = SummaryWriter()
            writer.add_mesh('my_mesh', vertices=vertices_tensor, colors=colors_tensor, faces=faces_tensor)

            writer.close()
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_mesh")
        self._get_file_writer().add_summary(
            mesh(tag, vertices, colors, faces, config_dict), global_step, walltime
        )

    def flush(self):
        """Flushes the event file to disk.

        Call this method to make sure that all pending events have been written to
        disk.
        """
        if self.all_writers is None:
            return
        for writer in self.all_writers.values():
            writer.flush()

    def close(self):
        if self.all_writers is None:
            return  # ignore double close
        for writer in self.all_writers.values():
            writer.flush()
            writer.close()
        self.file_writer = self.all_writers = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
