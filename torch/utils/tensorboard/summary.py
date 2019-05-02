from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import os
import re as _re

# pylint: disable=unused-import
from six.moves import range

from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.proto.summary_pb2 import HistogramProto
from tensorboard.compat.proto.summary_pb2 import SummaryMetadata
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.plugins.text.plugin_data_pb2 import TextPluginData
from tensorboard.plugins.pr_curve.plugin_data_pb2 import PrCurvePluginData
from tensorboard.plugins.custom_scalar import layout_pb2

from ._convert_np import make_np
from ._utils import _prepare_video, convert_to_HWC


_INVALID_TAG_CHARACTERS = _re.compile(r'[^-/\w\.]')


def _calc_scale_factor(tensor):
    converted = tensor.numpy() if not isinstance(tensor, np.ndarray) else tensor
    return 1 if converted.dtype == np.uint8 else 255


def _clean_tag(name):
    # In the past, the first argument to summary ops was a tag, which allowed
    # arbitrary characters. Now we are changing the first argument to be the node
    # name. This has a number of advantages (users of summary ops now can
    # take advantage of the tf name scope system) but risks breaking existing
    # usage, because a much smaller set of characters are allowed in node names.
    # This function replaces all illegal characters with _s, and logs a warning.
    # It also strips leading slashes from the name.
    if name is not None:
        new_name = _INVALID_TAG_CHARACTERS.sub('_', name)
        new_name = new_name.lstrip('/')  # Remove leading slashes
        if new_name != name:
            logging.info(
                'Summary name %s is illegal; using %s instead.' % (name, new_name))
            name = new_name
    return name


def _draw_single_box(image, xmin, ymin, xmax, ymax, display_str, color='black', color_text='black', thickness=2):
    from PIL import ImageDraw, ImageFont
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    if display_str:
        text_bottom = bottom
        # Reverse list and print from bottom to top.
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin),
             (left + text_width, text_bottom)], fill=color
        )
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str, fill=color_text, font=font
        )
    return image


def scalar(name, scalar, collections=None):
    """Outputs a `Summary` protocol buffer containing a single scalar value.
    The generated Summary has a Tensor.proto containing the input Tensor.
    Args:
      name: A name for the generated node. Will also serve as the series name in
        TensorBoard.
      tensor: A real numeric Tensor containing a single value.
      collections: Optional list of graph collections keys. The new summary op is
        added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    Returns:
      A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.
    Raises:
      ValueError: If tensor has the wrong shape or type.
    """
    name = _clean_tag(name)
    scalar = make_np(scalar)
    assert(scalar.squeeze().ndim == 0), 'scalar should be 0D'
    scalar = float(scalar)
    return Summary(value=[Summary.Value(tag=name, simple_value=scalar)])


def histogram_raw(name, min, max, num, sum, sum_squares, bucket_limits, bucket_counts):
    # pylint: disable=line-too-long
    """Outputs a `Summary` protocol buffer with a histogram.
    The generated
    [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
    has one summary value containing a histogram for `values`.
    Args:
      name: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      min: A float or int min value
      max: A float or int max value
      num: Int number of values
      sum: Float or int sum of all values
      sum_squares: Float or int sum of squares for all values
      bucket_limits: A numeric `Tensor` with upper value per bucket
      bucket_counts: A numeric `Tensor` with number of values per bucket
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    hist = HistogramProto(min=min,
                          max=max,
                          num=num,
                          sum=sum,
                          sum_squares=sum_squares,
                          bucket_limit=bucket_limits,
                          bucket=bucket_counts)
    return Summary(value=[Summary.Value(tag=name, histo=hist)])


def histogram(name, values, bins, max_bins=None):
    # pylint: disable=line-too-long
    """Outputs a `Summary` protocol buffer with a histogram.
    The generated
    [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
    has one summary value containing a histogram for `values`.
    This op reports an `InvalidArgument` error if any value is not finite.
    Args:
      name: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      values: A real numeric `Tensor`. Any shape. Values to use to
        build the histogram.
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    name = _clean_tag(name)
    values = make_np(values)
    hist = make_histogram(values.astype(float), bins, max_bins)
    return Summary(value=[Summary.Value(tag=name, histo=hist)])


def make_histogram(values, bins, max_bins=None):
    """Convert values into a histogram proto using logic from histogram.cc."""
    if values.size == 0:
        raise ValueError('The input has no element.')
    values = values.reshape(-1)
    counts, limits = np.histogram(values, bins=bins)
    num_bins = len(counts)
    if max_bins is not None and num_bins > max_bins:
        subsampling = num_bins // max_bins
        subsampling_remainder = num_bins % subsampling
        if subsampling_remainder != 0:
            counts = np.pad(counts, pad_width=[[0, subsampling - subsampling_remainder]],
                            mode="constant", constant_values=0)
        counts = counts.reshape(-1, subsampling).sum(axis=-1)
        new_limits = np.empty((counts.size + 1,), limits.dtype)
        new_limits[:-1] = limits[:-1:subsampling]
        new_limits[-1] = limits[-1]
        limits = new_limits

    # Find the first and the last bin defining the support of the histogram:
    cum_counts = np.cumsum(np.greater(counts, 0, dtype=np.int32))
    start, end = np.searchsorted(cum_counts, [0, cum_counts[-1] - 1], side="right")
    start = int(start)
    end = int(end) + 1
    del cum_counts

    # TensorBoard only includes the right bin limits. To still have the leftmost limit
    # included, we include an empty bin left.
    # If start == 0, we need to add an empty one left, otherwise we can just include the bin left to the
    # first nonzero-count bin:
    counts = counts[start - 1:end] if start > 0 else np.concatenate([[0], counts[:end]])
    limits = limits[start:end + 1]

    if counts.size == 0 or limits.size == 0:
        raise ValueError('The histogram is empty, please file a bug report.')

    sum_sq = values.dot(values)
    return HistogramProto(min=values.min(),
                          max=values.max(),
                          num=len(values),
                          sum=values.sum(),
                          sum_squares=sum_sq,
                          bucket_limit=limits.tolist(),
                          bucket=counts.tolist())


def image(tag, tensor, rescale=1, dataformats='NCHW'):
    """Outputs a `Summary` protocol buffer with images.
    The summary has up to `max_images` summary values containing images. The
    images are built from `tensor` which must be 3-D with shape `[height, width,
    channels]` and where `channels` can be:
    *  1: `tensor` is interpreted as Grayscale.
    *  3: `tensor` is interpreted as RGB.
    *  4: `tensor` is interpreted as RGBA.
    The `name` in the outputted Summary.Value protobufs is generated based on the
    name, with a suffix depending on the max_outputs setting:
    *  If `max_outputs` is 1, the summary value tag is '*name*/image'.
    *  If `max_outputs` is greater than 1, the summary value tags are
       generated sequentially as '*name*/image/0', '*name*/image/1', etc.
    Args:
      tag: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      tensor: A 3-D `uint8` or `float32` `Tensor` of shape `[height, width,
        channels]` where `channels` is 1, 3, or 4.
        'tensor' can either have values in [0, 1] (float32) or [0, 255] (uint8).
        The image() function will scale the image values to [0, 255] by applying
        a scale factor of either 1 (uint8) or 255 (float32).
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    tag = _clean_tag(tag)
    tensor = make_np(tensor)
    tensor = convert_to_HWC(tensor, dataformats)
    # Do not assume that user passes in values in [0, 255], use data type to detect
    scale_factor = _calc_scale_factor(tensor)
    tensor = tensor.astype(np.float32)
    tensor = (tensor * scale_factor).astype(np.uint8)
    image = make_image(tensor, rescale=rescale)
    return Summary(value=[Summary.Value(tag=tag, image=image)])


def image_boxes(tag, tensor_image, tensor_boxes, rescale=1, dataformats='CHW'):
    '''Outputs a `Summary` protocol buffer with images.'''
    tensor_image = make_np(tensor_image)
    tensor_image = convert_to_HWC(tensor_image, dataformats)
    tensor_boxes = make_np(tensor_boxes)
    tensor_image = tensor_image.astype(
        np.float32) * _calc_scale_factor(tensor_image)
    image = make_image(tensor_image.astype(np.uint8),
                       rescale=rescale,
                       rois=tensor_boxes)
    return Summary(value=[Summary.Value(tag=tag, image=image)])


def draw_boxes(disp_image, boxes):
    # xyxy format
    num_boxes = boxes.shape[0]
    list_gt = range(num_boxes)
    for i in list_gt:
        disp_image = _draw_single_box(disp_image,
                                      boxes[i, 0],
                                      boxes[i, 1],
                                      boxes[i, 2],
                                      boxes[i, 3],
                                      display_str=None,
                                      color='Red')
    return disp_image


def make_image(tensor, rescale=1, rois=None):
    """Convert an numpy representation image to Image protobuf"""
    from PIL import Image
    height, width, channel = tensor.shape
    scaled_height = int(height * rescale)
    scaled_width = int(width * rescale)
    image = Image.fromarray(tensor)
    if rois is not None:
        image = draw_boxes(image, rois)
    image = image.resize((scaled_width, scaled_height), Image.ANTIALIAS)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)


def video(tag, tensor, fps=4):
    tag = _clean_tag(tag)
    tensor = make_np(tensor)
    tensor = _prepare_video(tensor)
    # If user passes in uint8, then we don't need to rescale by 255
    scale_factor = _calc_scale_factor(tensor)
    tensor = tensor.astype(np.float32)
    tensor = (tensor * scale_factor).astype(np.uint8)
    video = make_video(tensor, fps)
    return Summary(value=[Summary.Value(tag=tag, image=video)])


def make_video(tensor, fps):
    try:
        import moviepy  # noqa: F401
    except ImportError:
        print('add_video needs package moviepy')
        return
    try:
        from moviepy import editor as mpy
    except ImportError:
        print("moviepy is installed, but can't import moviepy.editor.",
              "Some packages could be missing [imageio, requests]")
        return
    import tempfile

    t, h, w, c = tensor.shape

    # encode sequence of images into gif string
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)
    with tempfile.NamedTemporaryFile() as f:
        filename = f.name + '.gif'

    try:
        clip.write_gif(filename, verbose=False, progress_bar=False)
    except TypeError:
        clip.write_gif(filename, verbose=False)

    with open(filename, 'rb') as f:
        tensor_string = f.read()

    try:
        os.remove(filename)
    except OSError:
        pass

    return Summary.Image(height=h, width=w, colorspace=c, encoded_image_string=tensor_string)


def audio(tag, tensor, sample_rate=44100):
    tensor = make_np(tensor)
    tensor = tensor.squeeze()
    if abs(tensor).max() > 1:
        print('warning: audio amplitude out of range, auto clipped.')
        tensor = tensor.clip(-1, 1)
    assert(tensor.ndim == 1), 'input tensor should be 1 dimensional.'

    tensor_list = [int(32767.0 * x) for x in tensor]
    import io
    import wave
    import struct
    fio = io.BytesIO()
    Wave_write = wave.open(fio, 'wb')
    Wave_write.setnchannels(1)
    Wave_write.setsampwidth(2)
    Wave_write.setframerate(sample_rate)
    tensor_enc = b''
    for v in tensor_list:
        tensor_enc += struct.pack('<h', v)

    Wave_write.writeframes(tensor_enc)
    Wave_write.close()
    audio_string = fio.getvalue()
    fio.close()
    audio = Summary.Audio(sample_rate=sample_rate,
                          num_channels=1,
                          length_frames=len(tensor_list),
                          encoded_audio_string=audio_string,
                          content_type='audio/wav')
    return Summary(value=[Summary.Value(tag=tag, audio=audio)])


def custom_scalars(layout):
    categoriesnames = layout.keys()
    categories = []
    layouts = []
    for k, v in layout.items():
        charts = []
        for chart_name, chart_meatadata in v.items():
            tags = chart_meatadata[1]
            if chart_meatadata[0] == 'Margin':
                assert len(tags) == 3
                mgcc = layout_pb2.MarginChartContent(series=[layout_pb2.MarginChartContent.Series(value=tags[0],
                                                                                                  lower=tags[1],
                                                                                                  upper=tags[2])])
                chart = layout_pb2.Chart(title=chart_name, margin=mgcc)
            else:
                mlcc = layout_pb2.MultilineChartContent(tag=tags)
                chart = layout_pb2.Chart(title=chart_name, multiline=mlcc)
            charts.append(chart)
        categories.append(layout_pb2.Category(title=k, chart=charts))

    layout = layout_pb2.Layout(category=categories)
    PluginData = [SummaryMetadata.PluginData(plugin_name='custom_scalars')]
    smd = SummaryMetadata(plugin_data=PluginData)
    tensor = TensorProto(dtype='DT_STRING',
                         string_val=[layout.SerializeToString()],
                         tensor_shape=TensorShapeProto())
    return Summary(value=[Summary.Value(tag='custom_scalars__config__', tensor=tensor, metadata=smd)])


def text(tag, text):
    PluginData = SummaryMetadata.PluginData(
        plugin_name='text', content=TextPluginData(version=0).SerializeToString())
    smd = SummaryMetadata(plugin_data=PluginData)
    tensor = TensorProto(dtype='DT_STRING',
                         string_val=[text.encode(encoding='utf_8')],
                         tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]))
    return Summary(value=[Summary.Value(tag=tag + '/text_summary', metadata=smd, tensor=tensor)])


def pr_curve_raw(tag, tp, fp, tn, fn, precision, recall, num_thresholds=127, weights=None):
    if num_thresholds > 127:  # weird, value > 127 breaks protobuf
        num_thresholds = 127
    data = np.stack((tp, fp, tn, fn, precision, recall))
    pr_curve_plugin_data = PrCurvePluginData(
        version=0, num_thresholds=num_thresholds).SerializeToString()
    PluginData = SummaryMetadata.PluginData(
        plugin_name='pr_curves', content=pr_curve_plugin_data)
    smd = SummaryMetadata(plugin_data=PluginData)
    tensor = TensorProto(dtype='DT_FLOAT',
                         float_val=data.reshape(-1).tolist(),
                         tensor_shape=TensorShapeProto(
                             dim=[TensorShapeProto.Dim(size=data.shape[0]), TensorShapeProto.Dim(size=data.shape[1])]))
    return Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor)])


def pr_curve(tag, labels, predictions, num_thresholds=127, weights=None):
    # weird, value > 127 breaks protobuf
    num_thresholds = min(num_thresholds, 127)
    data = compute_curve(labels, predictions,
                         num_thresholds=num_thresholds, weights=weights)
    pr_curve_plugin_data = PrCurvePluginData(
        version=0, num_thresholds=num_thresholds).SerializeToString()
    PluginData = SummaryMetadata.PluginData(
        plugin_name='pr_curves', content=pr_curve_plugin_data)
    smd = SummaryMetadata(plugin_data=PluginData)
    tensor = TensorProto(dtype='DT_FLOAT',
                         float_val=data.reshape(-1).tolist(),
                         tensor_shape=TensorShapeProto(
                             dim=[TensorShapeProto.Dim(size=data.shape[0]), TensorShapeProto.Dim(size=data.shape[1])]))
    return Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor)])


# https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/summary.py
def compute_curve(labels, predictions, num_thresholds=None, weights=None):
    _MINIMUM_COUNT = 1e-7

    if weights is None:
        weights = 1.0

    # Compute bins of true positives and false positives.
    bucket_indices = np.int32(np.floor(predictions * (num_thresholds - 1)))
    float_labels = labels.astype(np.float)
    histogram_range = (0, num_thresholds - 1)
    tp_buckets, _ = np.histogram(
        bucket_indices,
        bins=num_thresholds,
        range=histogram_range,
        weights=float_labels * weights)
    fp_buckets, _ = np.histogram(
        bucket_indices,
        bins=num_thresholds,
        range=histogram_range,
        weights=(1.0 - float_labels) * weights)

    # Obtain the reverse cumulative sum.
    tp = np.cumsum(tp_buckets[::-1])[::-1]
    fp = np.cumsum(fp_buckets[::-1])[::-1]
    tn = fp[0] - fp
    fn = tp[0] - tp
    precision = tp / np.maximum(_MINIMUM_COUNT, tp + fp)
    recall = tp / np.maximum(_MINIMUM_COUNT, tp + fn)
    return np.stack((tp, fp, tn, fn, precision, recall))
