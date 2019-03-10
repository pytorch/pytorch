import gc
import numpy as np
import math
import json
import time

from .summary import compute_curve
from .utils import figure_to_image
from .x2num import make_np


# Decorator that checks if there is a Visdom connection
def _check_connection(fn):
    def wrapper(self, *args, **kwargs):
        if not self.server_connected:
            print('ERROR: No Visdom server currently connected')
            self._try_connect()
            return
        fn(self, *args, **kwargs)
    return wrapper


class VisdomWriter:
    def __init__(self):
        try:
            from visdom import Visdom
        except ImportError:
            raise ImportError(
                "Visdom visualization requires installation of Visdom")

        self.scalar_dict = {}
        self.server_connected = False
        self.vis = Visdom()
        self.windows = {}

        self._try_connect()

    def _try_connect(self):
        startup_sec = 1
        self.server_connected = self.vis.check_connection()
        while not self.server_connected and startup_sec > 0:
            time.sleep(0.1)
            startup_sec -= 0.1
            self.server_connected = self.vis.check_connection()
        assert self.server_connected, 'No connection could be formed quickly'

    @_check_connection
    def add_scalar(self, tag, scalar_value, global_step=None, main_tag='default'):
        """Add scalar data to Visdom. Plots the values in a plot titled
           {main_tag}-{tag}.

        Args:
            tag (string): Data identifier
            scalar_value (float or string/blobname): Value to save
            global_step (int): Global step value to record
            main_tag (string): Data group identifier
        """
        if self.scalar_dict.get(main_tag) is None:
            self.scalar_dict[main_tag] = {}
        exists = self.scalar_dict[main_tag].get(tag) is not None
        self.scalar_dict[main_tag][tag] = self.scalar_dict[main_tag][tag] + \
            [scalar_value] if exists else [scalar_value]
        plot_name = '{}-{}'.format(main_tag, tag)
        # If there is no global_step provided, follow sequential order
        x_val = len(self.scalar_dict[main_tag][tag]
                    ) if not global_step else global_step
        if exists:
            # Update our existing Visdom window
            self.vis.line(
                X=make_np(x_val),
                Y=make_np(scalar_value),
                name=plot_name,
                update='append',
                win=self.windows[plot_name],
            )
        else:
            # Save the window if we are creating this graph for the first time
            self.windows[plot_name] = self.vis.line(
                X=make_np(x_val),
                Y=make_np(scalar_value),
                name=plot_name,
                opts={
                    'title': plot_name,
                    'xlabel': 'timestep',
                    'ylabel': tag,
                },
            )

    @_check_connection
    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        """Adds many scalar data to summary.

        Note that this function also keeps logged scalars in memory. In extreme case it explodes your RAM.

        Args:
            tag (string): Data identifier
            main_tag (string): Data group identifier
            tag_scalar_dict (dict): Key-value pair storing the tag and corresponding values
            global_step (int): Global step value to record

        Examples::

            writer.add_scalars('run_14h',{'xsinx':i*np.sin(i/r),
                                          'xcosx':i*np.cos(i/r),
                                          'arctanx': numsteps*np.arctan(i/r)}, i)
            This function adds three plots:
                'run_14h-xsinx',
                'run_14h-xcosx',
                'run_14h-arctanx'
            with the corresponding values.
        """
        for key in tag_scalar_dict.keys():
            self.add_scalar(key, tag_scalar_dict[key], global_step, main_tag)

    @_check_connection
    def export_scalars_to_json(self, path):
        """Exports to the given 'path' an ASCII file containing all the scalars written
        so far by this instance, with the following format:
        {writer_id : [[timestamp, step, value], ...], ...}

        The scalars saved by ``add_scalars()`` will be flushed after export.
        """
        with open(path, "w") as f:
            json.dump(self.scalar_dict, f)
        self.scalar_dict = {}

    @_check_connection
    def add_histogram(self, tag, values, global_step=None, bins='tensorflow'):
        """Add histogram to summary.

        Args:
            tag (string): Data identifier
            values (torch.Tensor, numpy.array, or string/blobname): Values to build histogram
            global_step (int): Global step value to record
            bins (string): one of {'tensorflow', 'auto', 'fd', ...}, this determines how the bins are made. You can find
              other options in: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
        """
        values = make_np(values)
        self.vis.histogram(make_np(values), opts={'title': tag})

    @_check_connection
    def add_image(self, tag, img_tensor, global_step=None, caption=None):
        """Add image data to summary.

        Note that this requires the ``pillow`` package.

        Args:
            tag (string): Data identifier
            img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
            global_step (int): Global step value to record
        Shape:
            img_tensor: :math:`(C, H, W)`. Use ``torchvision.utils.make_grid()`` to prepare it is a good idea.
            C = colors (can be 1 - grayscale, 3 - RGB, 4 - RGBA)
        """
        img_tensor = make_np(img_tensor)
        self.vis.image(img_tensor, opts={'title': tag, 'caption': caption})

    @_check_connection
    def add_figure(self, tag, figure, global_step=None, close=True):
        """Render matplotlib figure into an image and add it to summary.

        Note that this requires the ``matplotlib`` package.

        Args:
            tag (string): Data identifier
            figure (matplotlib.pyplot.figure) or list of figures: figure or a list of figures
            global_step (int): Global step value to record
            close (bool): Flag to automatically close the figure
        """
        self.add_image(tag, figure_to_image(figure, close), global_step)

    @_check_connection
    def add_video(self, tag, vid_tensor, global_step=None, fps=4):
        """Add video data to summary.

        Note that this requires the ``moviepy`` package.

        Args:
            tag (string): Data identifier
            vid_tensor (torch.Tensor): Video data
            global_step (int): Global step value to record
            fps (float or int): Frames per second
        Shape:
            vid_tensor: :math:`(B, C, T, H, W)`. (if following tensorboardX format)
            vid_tensor: :math:`(T, H, W, C)`. (if following visdom format)
            B = batches, C = colors (1, 3, or 4), T = time frames, H = height, W = width
        """
        shape = vid_tensor.shape
        # A batch of videos (tensorboardX format) is a 5D tensor
        if len(shape) > 4:
            for i in range(shape[0]):
                # Reshape each video to Visdom's (T x H x W x C) and write each video
                # TODO: reverse the logic here, shoudl do the permutation in numpy
                if isinstance(vid_tensor, np.ndarray):
                    import torch
                    ind_vid = torch.from_numpy(
                        vid_tensor[i, :, :, :, :]).permute(1, 2, 3, 0)
                else:
                    ind_vid = vid_tensor[i, :, :, :, :].permute(1, 2, 3, 0)
                scale_factor = 255 if np.any(
                    (ind_vid > 0) & (ind_vid < 1)) else 1
                # Visdom looks for .ndim attr, this is something raw Tensors don't have
                # Cast to Numpy array to get .ndim attr
                ind_vid = ind_vid.numpy()
                ind_vid = (ind_vid * scale_factor).astype(np.uint8)
                assert ind_vid.shape[3] in [1, 3, 4], \
                    'Visdom requires the last dimension to be color, which can be 1 (grayscale), 3 (RGB) or 4 (RGBA)'
                self.vis.video(tensor=ind_vid, opts={'fps': fps})
        else:
            self.vis.video(tensor=vid_tensor, opts={'fps': fps})

    @_check_connection
    def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100):
        """Add audio data to summary.

        Args:
            tag (string): Data identifier
            snd_tensor (torch.Tensor, numpy.array, or string/blobname): Sound data
            global_step (int): Global step value to record
            sample_rate (int): sample rate in Hz

        Shape:
            snd_tensor: :math:`(1, L)`. The values should lie between [-1, 1].
        """
        snd_tensor = make_np(snd_tensor)
        self.vis.audio(tensor=snd_tensor, opts={
                       'sample_frequency': sample_rate})

    @_check_connection
    def add_text(self, tag, text_string, global_step=None):
        """Add text data to summary.

        Args:
            tag (string): Data identifier
            text_string (string): String to save
            global_step (int): Global step value to record
        Examples::
            writer.add_text('lstm', 'This is an lstm', 0)
            writer.add_text('rnn', 'This is an rnn', 10)
        """
        if text_string is None:
            # Visdom doesn't support tags, write the tag as the text_string
            text_string = tag
        self.vis.text(text_string)

    @_check_connection
    def add_onnx_graph(self, prototxt):
        # TODO: Visdom doesn't support graph visualization yet, so this is a no-op
        return

    @_check_connection
    def add_graph(self, model, input_to_model=None, verbose=False, **kwargs):
        # TODO: Visdom doesn't support graph visualization yet, so this is a no-op
        return

    @_check_connection
    def add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None):
        # TODO: Visdom doesn't support embeddings yet, so this is a no-op
        return

    @_check_connection
    def add_pr_curve(self, tag, labels, predictions, global_step=None, num_thresholds=127, weights=None):
        """Adds precision recall curve.

        Args:
            tag (string): Data identifier
            labels (torch.Tensor, numpy.array, or string/blobname): Ground truth data. Binary label for each element.
            predictions (torch.Tensor, numpy.array, or string/blobname):
            The probability that an element be classified as true. Value should in [0, 1]
            global_step (int): Global step value to record
            num_thresholds (int): Number of thresholds used to draw the curve.

        """
        labels, predictions = make_np(labels), make_np(predictions)
        raw_data = compute_curve(labels, predictions, num_thresholds, weights)

        # compute_curve returns np.stack((tp, fp, tn, fn, precision, recall))
        # We want to access 'precision' and 'recall'
        precision, recall = raw_data[4, :], raw_data[5, :]

        self.vis.line(
            X=recall,
            Y=precision,
            name=tag,
            opts={
                'title': 'PR Curve for {}'.format(tag),
                'xlabel': 'recall',
                'ylabel': 'precision',
            },
        )

    @_check_connection
    def add_pr_curve_raw(self, tag, true_positive_counts,
                         false_positive_counts,
                         true_negative_counts,
                         false_negative_counts,
                         precision,
                         recall, global_step=None, num_thresholds=127, weights=None):
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
            see: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/README.md
        """
        precision, recall = make_np(precision), make_np(recall)
        self.vis.line(
            X=recall,
            Y=precision,
            name=tag,
            opts={
                'title': 'PR Curve for {}'.format(tag),
                'xlabel': 'recall',
                'ylabel': 'precision',
            },
        )

    def close(self):
        del self.vis
        del self.scalar_dict
        gc.collect()
