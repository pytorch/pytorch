# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides an API for generating Event protocol buffers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import six
import time

from .embedding import make_mat, make_sprite, make_tsv, append_pbtxt
from .event_file_writer import EventFileWriter
from .onnx_graph import gg
from .pytorch_graph import graph
from .proto import event_pb2
from .proto import summary_pb2
from .proto import graph_pb2
from .summary import scalar, histogram, image, audio, text, pr_curve, pr_curve_raw, video, custom_scalars
from .utils import figure_to_image
from tensorboardX.proto.event_pb2 import SessionLog
from tensorboardX.proto.event_pb2 import Event
from tensorboardX.summary import image_boxes


class SummaryToEventTransformer(object):
    """Abstractly implements the SummaryWriter API.
    This API basically implements a number of endpoints (add_summary,
    add_session_log, etc). The endpoints all generate an event protobuf, which is
    passed to the contained event_writer.
    @@__init__
    @@add_summary
    @@add_session_log
    @@add_graph
    @@add_meta_graph
    @@add_run_metadata
    """

    def __init__(self, event_writer, graph=None, graph_def=None):
        """Creates a `SummaryWriter` and an event file.
        On construction the summary writer creates a new event file in `logdir`.
        This event file will contain `Event` protocol buffers constructed when you
        call one of the following functions: `add_summary()`, `add_session_log()`,
        `add_event()`, or `add_graph()`.
        If you pass a `Graph` to the constructor it is added to
        the event file. (This is equivalent to calling `add_graph()` later).
        TensorBoard will pick the graph from the file and display it graphically so
        you can interactively explore the graph you built. You will usually pass
        the graph from the session in which you launched it:
        ```python
        ...create a graph...
        # Launch the graph in a session.
        sess = tf.Session()
        # Create a summary writer, add the 'graph' to the event file.
        writer = tf.summary.FileWriter(<some-directory>, sess.graph)
        ```
        Args:
          event_writer: An EventWriter. Implements add_event method.
          graph: A `Graph` object, such as `sess.graph`.
          graph_def: DEPRECATED: Use the `graph` argument instead.
        """
        self.event_writer = event_writer
        # For storing used tags for session.run() outputs.
        self._session_run_tags = {}
        # TODO(zihaolucky). pass this an empty graph to check whether it's necessary.
        # currently we don't support graph in MXNet using tensorboard.

    def add_summary(self, summary, global_step=None, walltime=None):
        """Adds a `Summary` protocol buffer to the event file.
        This method wraps the provided summary in an `Event` protocol buffer
        and adds it to the event file.
        You can pass the result of evaluating any summary op, using
        [`Session.run()`](client.md#Session.run) or
        [`Tensor.eval()`](framework.md#Tensor.eval), to this
        function. Alternatively, you can pass a `tf.Summary` protocol
        buffer that you populate with your own data. The latter is
        commonly done to report evaluation results in event files.
        Args:
          summary: A `Summary` protocol buffer, optionally serialized as a string.
          global_step: Number. Optional global step value to record with the
            summary.
          walltime: float. Optional walltime to override the default (current)
            walltime (from time.time())
        """
        if isinstance(summary, bytes):
            summ = summary_pb2.Summary()
            summ.ParseFromString(summary)
            summary = summ
        event = event_pb2.Event(summary=summary)
        self._add_event(event, global_step, walltime)

    def add_graph(self, graph_profile, walltime=None):
        graph = graph_profile[0]
        stepstats = graph_profile[1]
        """Adds a `Graph` protocol buffer to the event file.
        """
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self._add_event(event, None, walltime)

        trm = event_pb2.TaggedRunMetadata(
            tag='step1', run_metadata=stepstats.SerializeToString())
        event = event_pb2.Event(tagged_run_metadata=trm)
        self._add_event(event, None, walltime)

    def add_onnx_graph(self, graph, walltime=None):
        """Adds a `Graph` protocol buffer to the event file.
        """
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self._add_event(event, None, walltime)

    def add_session_log(self, session_log, global_step=None, walltime=None):
        """Adds a `SessionLog` protocol buffer to the event file.
        This method wraps the provided session in an `Event` protocol buffer
        and adds it to the event file.
        Args:
          session_log: A `SessionLog` protocol buffer.
          global_step: Number. Optional global step value to record with the
            summary.
        """
        event = event_pb2.Event(session_log=session_log)
        self._add_event(event, global_step, walltime)

    def _add_event(self, event, step, walltime):
        event.wall_time = time.time() if walltime is None else walltime
        if step is not None:
            event.step = int(step)
        self.event_writer.add_event(event)


class FileWriter(SummaryToEventTransformer):
    """Writes `Summary` protocol buffers to event files.
    The `FileWriter` class provides a mechanism to create an event file in a
    given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    @@__init__
    @@add_summary
    @@add_session_log
    @@add_event
    @@add_graph
    @@add_run_metadata
    @@get_logdir
    @@flush
    @@close
    """

    def __init__(self,
                 logdir,
                 graph=None,
                 max_queue=10,
                 flush_secs=120,
                 filename_suffix='',
                 graph_def=None):
        """Creates a `FileWriter` and an event file.
        On construction the summary writer creates a new event file in `logdir`.
        This event file will contain `Event` protocol buffers constructed when you
        call one of the following functions: `add_summary()`, `add_session_log()`,
        `add_event()`, or `add_graph()`.
        If you pass a `Graph` to the constructor it is added to
        the event file. (This is equivalent to calling `add_graph()` later).
        TensorBoard will pick the graph from the file and display it graphically so
        you can interactively explore the graph you built. You will usually pass
        the graph from the session in which you launched it:
        ```python
        ...create a graph...
        # Launch the graph in a session.
        sess = tf.Session()
        # Create a summary writer, add the 'graph' to the event file.
        writer = tf.summary.FileWriter(<some-directory>, sess.graph)
        ```
        The other arguments to the constructor control the asynchronous writes to
        the event file:
        *  `flush_secs`: How often, in seconds, to flush the added summaries
           and events to disk.
        *  `max_queue`: Maximum number of summaries or events pending to be
           written to disk before one of the 'add' calls block.
        Args:
          logdir: A string. Directory where event file will be written.
          graph: A `Graph` object, such as `sess.graph`.
          max_queue: Integer. Size of the queue for pending events and summaries.
          flush_secs: Number. How often, in seconds, to flush the
            pending events and summaries to disk.
          graph_def: DEPRECATED: Use the `graph` argument instead.
        """
        logdir = str(logdir)
        event_writer = EventFileWriter(
            logdir, max_queue, flush_secs, filename_suffix)
        super(FileWriter, self).__init__(event_writer, graph, graph_def)

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self.event_writer.get_logdir()

    def add_event(self, event):
        """Adds an event to the event file.
        Args:
          event: An `Event` protocol buffer.
        """
        self.event_writer.add_event(event)

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


class SummaryWriter(object):
    """Writes `Summary` directly to event files.
    The `SummaryWriter` class provides a high-level api to create an event file in a
    given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """

    def __init__(self, log_dir=None, comment='', **kwargs):
        """
        Args:
            log_dir (string): save location, default is: runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each
              run. Use hierarchical folder structure to compare between runs easily. e.g. 'runs/exp1', 'runs/exp2'
            comment (string): comment that appends to the default ``log_dir``. If ``log_dir`` is assigned,
              this argument will no effect.
            purge_step (int):
              When logging crashes at step :math:`T+X` and restarts at step :math:`T`, any events
              whose global_step larger or equal to :math:`T` will be purged and hidden from TensorBoard.
              Note that the resumed experiment and crashed experiment should have the same ``log_dir``.
            filename_suffix (string):
              Every event file's name is suffixed with suffix. example: ``SummaryWriter(filename_suffix='.123')``
            kwargs: extra keyword arguments for FileWriter (e.g. 'flush_secs'
              controls how often to flush pending events). For more arguments
              please refer to docs for 'tf.summary.FileWriter'.
        """
        if not log_dir:
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join(
                'runs', current_time + '_' + socket.gethostname() + comment)
        self.log_dir = log_dir

        if 'purge_step' in kwargs.keys():
            most_recent_step = kwargs.pop('purge_step')
            self.file_writer = FileWriter(logdir=log_dir, **kwargs)
            self.file_writer.add_event(
                Event(step=most_recent_step, file_version='brain.Event:2'))
            self.file_writer.add_event(
                Event(step=most_recent_step, session_log=SessionLog(status=SessionLog.START)))
        else:
            self.file_writer = FileWriter(logdir=log_dir, **kwargs)

        # Create default bins for histograms, see generate_testdata.py in tensorflow/tensorboard
        v = 1E-12
        buckets = []
        neg_buckets = []
        while v < 1E20:
            buckets.append(v)
            neg_buckets.append(-v)
            v *= 1.1
        self.default_bins = neg_buckets[::-1] + [0] + buckets

        self.all_writers = {self.file_writer.get_logdir(): self.file_writer}
        # {writer_id : [[timestamp, step, value],...],...}
        self.scalar_dict = {}

        # TODO (ml7): Remove try-except when PyTorch 1.0 merges PyTorch and Caffe2
        try:
            import caffe2
            from caffe2.python import workspace  # workaround for pytorch/issue#10249
            self.caffe2_enabled = True
        except (SystemExit, ImportError):
            self.caffe2_enabled = False

    def __append_to_scalar_dict(self, tag, scalar_value, global_step,
                                timestamp):
        """This adds an entry to the self.scalar_dict datastructure with format
        {writer_id : [[timestamp, step, value], ...], ...}.
        """
        from .x2num import make_np
        if tag not in self.scalar_dict.keys():
            self.scalar_dict[tag] = []
        self.scalar_dict[tag].append(
            [timestamp, global_step, float(make_np(scalar_value))])

    def _check_caffe2(self, item):
        """
        Caffe2 users have the option of passing a string representing the name of
        a blob in the workspace instead of passing the actual Tensor/array containing
        the numeric values. Thus, we need to check if we received a string as input
        instead of an actual Tensor/array, and if so, we need to fetch the Blob
        from the workspace corresponding to that name. Fetching can be done with the
        following:

        from caffe2.python import workspace (if not already imported)
        workspace.FetchBlob(blob_name)
        workspace.FetchBlobs([blob_name1, blob_name2, ...])
        """
        # TODO (ml7): Remove caffe2_enabled check when PyTorch 1.0 merges PyTorch and Caffe2
        return self.caffe2_enabled and isinstance(item, six.string_types)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        """Add scalar data to summary.

        Args:
            tag (string): Data identifier
            scalar_value (float or string/blobname): Value to save
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time()) of event
        """
        if self._check_caffe2(scalar_value):
            scalar_value = workspace.FetchBlob(scalar_value)
        self.file_writer.add_summary(
            scalar(tag, scalar_value), global_step, walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        """Adds many scalar data to summary.

        Note that this function also keeps logged scalars in memory. In extreme case it explodes your RAM.

        Args:
            main_tag (string): The parent name for the tags
            tag_scalar_dict (dict): Key-value pair storing the tag and corresponding values
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time()) of event

        Examples::

            writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                           'xcosx':i*np.cos(i/r),
                                           'arctanx': numsteps*np.arctan(i/r)}, i)
            # This call adds three values to the same scalar plot with the tag
            # 'run_14h' in TensorBoard's scalar section.
        """
        walltime = time.time() if walltime is None else walltime
        fw_logdir = self.file_writer.get_logdir()
        for tag, scalar_value in tag_scalar_dict.items():
            fw_tag = fw_logdir + "/" + main_tag + "/" + tag
            if fw_tag in self.all_writers.keys():
                fw = self.all_writers[fw_tag]
            else:
                fw = FileWriter(logdir=fw_tag)
                self.all_writers[fw_tag] = fw
            if self._check_caffe2(scalar_value):
                scalar_value = workspace.FetchBlob(scalar_value)
            fw.add_summary(scalar(main_tag, scalar_value),
                           global_step, walltime)
            self.__append_to_scalar_dict(
                fw_tag, scalar_value, global_step, walltime)

    def export_scalars_to_json(self, path):
        """Exports to the given path an ASCII file containing all the scalars written
        so far by this instance, with the following format:
        {writer_id : [[timestamp, step, value], ...], ...}

        The scalars saved by ``add_scalars()`` will be flushed after export.
        """
        with open(path, "w") as f:
            json.dump(self.scalar_dict, f)
        self.scalar_dict = {}

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None):
        """Add histogram to summary.

        Args:
            tag (string): Data identifier
            values (torch.Tensor, numpy.array, or string/blobname): Values to build histogram
            global_step (int): Global step value to record
            bins (string): one of {'tensorflow','auto', 'fd', ...}, this determines how the bins are made. You can find
              other options in: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
            walltime (float): Optional override default walltime (time.time()) of event
        """
        if self._check_caffe2(values):
            values = workspace.FetchBlob(values)
        if isinstance(bins, six.string_types) and bins == 'tensorflow':
            bins = self.default_bins
        self.file_writer.add_summary(
            histogram(tag, values, bins, max_bins=max_bins), global_step, walltime)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        """Add image data to summary.

        Note that this requires the ``pillow`` package.

        Args:
            tag (string): Data identifier
            img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time()) of event
        Shape:
            img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
            convert a batch of tensor into 3xHxW format or call ``add_images`` and let tensorboardX do the job.
            Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitible as long as
            corresponding ``dataformats`` argument is passed. e.g. CHW, HWC, HW.
        """
        if self._check_caffe2(img_tensor):
            img_tensor = workspace.FetchBlob(img_tensor)
        self.file_writer.add_summary(
            image(tag, img_tensor, dataformats=dataformats), global_step, walltime)

    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
        """Add batched image data to summary.

        Note that this requires the ``pillow`` package.

        Args:
            tag (string): Data identifier
            img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time()) of event
        Shape:
            img_tensor: Default is :math:`(N, 3, H, W)`. If ``dataformats`` is specified, other shape will be
            accepted. e.g. NCHW or NHWC.
        """
        if self._check_caffe2(img_tensor):
            img_tensor = workspace.FetchBlob(img_tensor)
        self.file_writer.add_summary(
            image(tag, img_tensor, dataformats=dataformats), global_step, walltime)

    def add_image_with_boxes(self, tag, img_tensor, box_tensor, global_step=None,
                             walltime=None, dataformats='CHW', **kwargs):
        """Add image and draw bounding boxes on the image.

        Args:
            tag (string): Data identifier
            img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
            box_tensor (torch.Tensor, numpy.array, or string/blobname): Box data (for detected objects)
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time()) of event
        Shape:
            img_tensor: Default is :math:`(3, H, W)`. It can be specified with ``dataformat`` agrument.
            e.g. CHW or HWC

            box_tensor: (torch.Tensor, numpy.array, or string/blobname): NX4,  where N is the number of
            boxes and each 4 elememts in a row represents (xmin, ymin, xmax, ymax).
        """
        if self._check_caffe2(img_tensor):
            img_tensor = workspace.FetchBlob(img_tensor)
        if self._check_caffe2(box_tensor):
            box_tensor = workspace.FetchBlob(box_tensor)
        self.file_writer.add_summary(image_boxes(
            tag, img_tensor, box_tensor, dataformats=dataformats, **kwargs), global_step, walltime)

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        """Render matplotlib figure into an image and add it to summary.

        Note that this requires the ``matplotlib`` package.

        Args:
            tag (string): Data identifier
            figure (matplotlib.pyplot.figure) or list of figures: figure or a list of figures
            global_step (int): Global step value to record
            close (bool): Flag to automatically close the figure
            walltime (float): Optional override default walltime (time.time()) of event
        """
        if isinstance(figure, list):
            self.add_image(tag, figure_to_image(figure, close), global_step, walltime, dataformats='NCHW')
        else:
            self.add_image(tag, figure_to_image(figure, close), global_step, walltime, dataformats='CHW')

    def add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None):
        """Add video data to summary.

        Note that this requires the ``moviepy`` package.

        Args:
            tag (string): Data identifier
            vid_tensor (torch.Tensor): Video data
            global_step (int): Global step value to record
            fps (float or int): Frames per second
            walltime (float): Optional override default walltime (time.time()) of event
        Shape:
            vid_tensor: :math:`(N, T, C, H, W)`.
        """
        self.file_writer.add_summary(
            video(tag, vid_tensor, fps), global_step, walltime)

    def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None):
        """Add audio data to summary.

        Args:
            tag (string): Data identifier
            snd_tensor (torch.Tensor): Sound data
            global_step (int): Global step value to record
            sample_rate (int): sample rate in Hz
            walltime (float): Optional override default walltime (time.time()) of event
        Shape:
            snd_tensor: :math:`(1, L)`. The values should lie between [-1, 1].
        """
        if self._check_caffe2(snd_tensor):
            snd_tensor = workspace.FetchBlob(snd_tensor)
        self.file_writer.add_summary(
            audio(tag, snd_tensor, sample_rate=sample_rate), global_step, walltime)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        """Add text data to summary.

        Args:
            tag (string): Data identifier
            text_string (string): String to save
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time()) of event
        Examples::

            writer.add_text('lstm', 'This is an lstm', 0)
            writer.add_text('rnn', 'This is an rnn', 10)
        """
        self.file_writer.add_summary(
            text(tag, text_string), global_step, walltime)

    def add_onnx_graph(self, prototxt):
        self.file_writer.add_onnx_graph(gg(prototxt))

    def add_graph(self, model, input_to_model=None, verbose=False, **kwargs):
        # prohibit second call?
        # no, let tensorboard handle it and show its warning message.
        """Add graph data to summary.

        Args:
            model (torch.nn.Module): model to draw.
            input_to_model (torch.Tensor or list of torch.Tensor): a variable or a tuple of
                variables to be fed.

        """
        if hasattr(model, 'forward'):
            # A valid PyTorch model should have a 'forward' method
            import torch
            from distutils.version import LooseVersion
            if LooseVersion(torch.__version__) >= LooseVersion("0.3.1"):
                pass
            else:
                if LooseVersion(torch.__version__) >= LooseVersion("0.3.0"):
                    print('You are using PyTorch==0.3.0, use add_onnx_graph()')
                    return
                if not hasattr(torch.autograd.Variable, 'grad_fn'):
                    print('add_graph() only supports PyTorch v0.2.')
                    return
            self.file_writer.add_graph(graph(model, input_to_model, verbose))
        else:
            # Caffe2 models do not have the 'forward' method
            if not self.caffe2_enabled:
                # TODO (ml7): Remove when PyTorch 1.0 merges PyTorch and Caffe2
                return
            from caffe2.proto import caffe2_pb2
            from caffe2.python import core
            from .caffe2_graph import (
                model_to_graph_def, nets_to_graph_def, protos_to_graph_def
            )
            # notimporterror should be already handled when checking self.caffe2_enabled

            '''Write graph to the summary. Check model type and handle accordingly.'''
            if isinstance(model, list):
                if isinstance(model[0], core.Net):
                    current_graph = nets_to_graph_def(
                        model, **kwargs)
                elif isinstance(model[0], caffe2_pb2.NetDef):
                    current_graph = protos_to_graph_def(
                        model, **kwargs)
            # Handles cnn.CNNModelHelper, model_helper.ModelHelper
            else:
                current_graph = model_to_graph_def(
                    model, **kwargs)
            event = event_pb2.Event(
                graph_def=current_graph.SerializeToString())
            self.file_writer.add_event(event)

    @staticmethod
    def _encode(rawstr):
        # I'd use urllib but, I'm unsure about the differences from python3 to python2, etc.
        retval = rawstr
        retval = retval.replace("%", "%%%02x" % (ord("%")))
        retval = retval.replace("/", "%%%02x" % (ord("/")))
        retval = retval.replace("\\", "%%%02x" % (ord("\\")))
        return retval

    def add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None):
        """Add embedding projector data to summary.

        Args:
            mat (torch.Tensor or numpy.array): A matrix which each row is the feature vector of the data point
            metadata (list): A list of labels, each element will be convert to string
            label_img (torch.Tensor): Images correspond to each data point
            global_step (int): Global step value to record
            tag (string): Name for the embedding
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
        from .x2num import make_np
        mat = make_np(mat)
        if global_step is None:
            global_step = 0
            # clear pbtxt?
        # Maybe we should encode the tag so slashes don't trip us up?
        # I don't think this will mess us up, but better safe than sorry.
        subdir = "%s/%s" % (str(global_step).zfill(5), self._encode(tag))
        save_path = os.path.join(self.file_writer.get_logdir(), subdir)
        try:
            os.makedirs(save_path)
        except OSError:
            print(
                'warning: Embedding dir exists, did you set global_step for add_embedding()?')
        if metadata is not None:
            assert mat.shape[0] == len(
                metadata), '#labels should equal with #data points'
            make_tsv(metadata, save_path, metadata_header=metadata_header)
        if label_img is not None:
            assert mat.shape[0] == label_img.shape[0], '#images should equal with #data points'
            make_sprite(label_img, save_path)
        assert mat.ndim == 2, 'mat should be 2D, where mat.size(0) is the number of data points'
        make_mat(mat, save_path)
        # new funcion to append to the config file a new embedding
        append_pbtxt(metadata, label_img,
                     self.file_writer.get_logdir(), subdir, global_step, tag)

    def add_pr_curve(self, tag, labels, predictions, global_step=None,
                     num_thresholds=127, weights=None, walltime=None):
        """Adds precision recall curve.

        Args:
            tag (string): Data identifier
            labels (torch.Tensor, numpy.array, or string/blobname): Ground truth data. Binary label for each element.
            predictions (torch.Tensor, numpy.array, or string/blobname):
            The probability that an element be classified as true. Value should in [0, 1]
            global_step (int): Global step value to record
            num_thresholds (int): Number of thresholds used to draw the curve.
            walltime (float): Optional override default walltime (time.time()) of event

        """
        from .x2num import make_np
        labels, predictions = make_np(labels), make_np(predictions)
        self.file_writer.add_summary(
            pr_curve(tag, labels, predictions, num_thresholds, weights),
            global_step, walltime)

    def add_pr_curve_raw(self, tag, true_positive_counts,
                         false_positive_counts,
                         true_negative_counts,
                         false_negative_counts,
                         precision,
                         recall,
                         global_step=None,
                         num_thresholds=127,
                         weights=None,
                         walltime=None):
        """Adds precision recall curve with raw data.

        Args:
            tag (string): Data identifier
            true_positive_counts (torch.Tensor, numpy.array, or string/blobname): true positive counts
            false_positive_counts (torch.Tensor, numpy.array, or string/blobname): false positive counts
            true_negative_counts (torch.Tensor, numpy.array, or string/blobname): true negative counts
            false_negative_counts (torch.Tensor, numpy.array, or string/blobname): false negative counts
            precision (torch.Tensor, numpy.array, or string/blobname): precision
            recall (torch.Tensor, numpy.array, or string/blobname): recall
            global_step (int): Global step value to record
            num_thresholds (int): Number of thresholds used to draw the curve.
            walltime (float): Optional override default walltime (time.time()) of event
            see: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/README.md
        """
        self.file_writer.add_summary(
            pr_curve_raw(tag,
                         true_positive_counts,
                         false_positive_counts,
                         true_negative_counts,
                         false_negative_counts,
                         precision,
                         recall,
                         num_thresholds,
                         weights),
            global_step,
            walltime)

    def add_custom_scalars_multilinechart(self, tags, category='default', title='untitled'):
        """Shorthand for creating multilinechart. Similar to ``add_custom_scalars()``, but the only necessary argument
        is *tags*.

        Args:
            tags (list): list of tags that have been used in ``add_scalar()``

        Examples::

            writer.add_custom_scalars_multilinechart(['twse/0050', 'twse/2330'])
        """
        layout = {category: {title: ['Multiline', tags]}}
        self.file_writer.add_summary(custom_scalars(layout))

    def add_custom_scalars_marginchart(self, tags, category='default', title='untitled'):
        """Shorthand for creating marginchart. Similar to ``add_custom_scalars()``, but the only necessary argument
        is *tags*, which should have exactly 3 elements.

        Args:
            tags (list): list of tags that have been used in ``add_scalar()``

        Examples::

            writer.add_custom_scalars_marginchart(['twse/0050', 'twse/2330', 'twse/2006'])
        """
        assert len(tags) == 3
        layout = {category: {title: ['Margin', tags]}}
        self.file_writer.add_summary(custom_scalars(layout))

    def add_custom_scalars(self, layout):
        """Create special chart by collecting charts tags in 'scalars'. Note that this function can only be called once
        for each SummaryWriter() object. Because it only provides metadata to tensorboard, the function can be called
        before or after the training loop. See ``examples/demo_custom_scalars.py`` for more.

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
        self.file_writer.add_summary(custom_scalars(layout))

    def close(self):
        if self.file_writer is None:
            return  # ignore double close
        for path, writer in self.all_writers.items():
            writer.flush()
            writer.close()
        self.file_writer = self.all_writers = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
