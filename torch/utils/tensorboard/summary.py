import json
import logging
import os
from typing import Optional

import numpy as np
from google.protobuf import struct_pb2

# pylint: disable=unused-import
from six.moves import range
from tensorboard.compat.proto.summary_pb2 import HistogramProto
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.proto.summary_pb2 import SummaryMetadata
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.plugins.pr_curve.plugin_data_pb2 import PrCurvePluginData
from tensorboard.plugins.text.plugin_data_pb2 import TextPluginData

from ._convert_np import make_np
from ._utils import _prepare_video, convert_to_HWC

__all__ = ['hparams', 'scalar', 'histogram_raw', 'histogram', 'make_histogram', 'image', 'image_boxes', 'draw_boxes',
           'make_image', 'video', 'make_video', 'audio', 'custom_scalars', 'text', 'pr_curve_raw', 'pr_curve', 'compute_curve',
           'mesh']

logger = logging.getLogger(__name__)


def _calc_scale_factor(tensor):
    converted = tensor.numpy() if not isinstance(tensor, np.ndarray) else tensor
    return 1 if converted.dtype == np.uint8 else 255


def _draw_single_box(
    image,
    xmin,
    ymin,
    xmax,
    ymax,
    display_str,
    color="black",
    color_text="black",
    thickness=2,
):
    from PIL import ImageDraw, ImageFont

    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=color,
    )
    if display_str:
        text_bottom = bottom
        # Reverse list and print from bottom to top.
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom),
            ],
            fill=color,
        )
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill=color_text,
            font=font,
        )
    return image


def hparams(hparam_dict=None, metric_dict=None, hparam_domain_discrete=None):
    """Outputs three `Summary` protocol buffers needed by hparams plugin.
    `Experiment` keeps the metadata of an experiment, such as the name of the
      hyperparameters and the name of the metrics.
    `SessionStartInfo` keeps key-value pairs of the hyperparameters
    `SessionEndInfo` describes status of the experiment e.g. STATUS_SUCCESS

    Args:
      hparam_dict: A dictionary that contains names of the hyperparameters
        and their values.
      metric_dict: A dictionary that contains names of the metrics
        and their values.
      hparam_domain_discrete: (Optional[Dict[str, List[Any]]]) A dictionary that
        contains names of the hyperparameters and all discrete values they can hold

    Returns:
      The `Summary` protobufs for Experiment, SessionStartInfo and
        SessionEndInfo
    """
    import torch
    from six import string_types
    from tensorboard.plugins.hparams.api_pb2 import (
        Experiment,
        HParamInfo,
        MetricInfo,
        MetricName,
        Status,
        DataType,
    )
    from tensorboard.plugins.hparams.metadata import (
        PLUGIN_NAME,
        PLUGIN_DATA_VERSION,
        EXPERIMENT_TAG,
        SESSION_START_INFO_TAG,
        SESSION_END_INFO_TAG,
    )
    from tensorboard.plugins.hparams.plugin_data_pb2 import (
        HParamsPluginData,
        SessionEndInfo,
        SessionStartInfo,
    )

    # TODO: expose other parameters in the future.
    # hp = HParamInfo(name='lr',display_name='learning rate',
    # type=DataType.DATA_TYPE_FLOAT64, domain_interval=Interval(min_value=10,
    # max_value=100))
    # mt = MetricInfo(name=MetricName(tag='accuracy'), display_name='accuracy',
    # description='', dataset_type=DatasetType.DATASET_VALIDATION)
    # exp = Experiment(name='123', description='456', time_created_secs=100.0,
    # hparam_infos=[hp], metric_infos=[mt], user='tw')

    if not isinstance(hparam_dict, dict):
        logger.warning("parameter: hparam_dict should be a dictionary, nothing logged.")
        raise TypeError(
            "parameter: hparam_dict should be a dictionary, nothing logged."
        )
    if not isinstance(metric_dict, dict):
        logger.warning("parameter: metric_dict should be a dictionary, nothing logged.")
        raise TypeError(
            "parameter: metric_dict should be a dictionary, nothing logged."
        )

    hparam_domain_discrete = hparam_domain_discrete or {}
    if not isinstance(hparam_domain_discrete, dict):
        raise TypeError(
            "parameter: hparam_domain_discrete should be a dictionary, nothing logged."
        )
    for k, v in hparam_domain_discrete.items():
        if (
            k not in hparam_dict
            or not isinstance(v, list)
            or not all(isinstance(d, type(hparam_dict[k])) for d in v)
        ):
            raise TypeError(
                "parameter: hparam_domain_discrete[{}] should be a list of same type as "
                "hparam_dict[{}].".format(k, k)
            )
    hps = []

    ssi = SessionStartInfo()
    for k, v in hparam_dict.items():
        if v is None:
            continue
        if isinstance(v, int) or isinstance(v, float):
            ssi.hparams[k].number_value = v

            if k in hparam_domain_discrete:
                domain_discrete: Optional[struct_pb2.ListValue] = struct_pb2.ListValue(
                    values=[
                        struct_pb2.Value(number_value=d)
                        for d in hparam_domain_discrete[k]
                    ]
                )
            else:
                domain_discrete = None

            hps.append(
                HParamInfo(
                    name=k,
                    type=DataType.Value("DATA_TYPE_FLOAT64"),
                    domain_discrete=domain_discrete,
                )
            )
            continue

        if isinstance(v, string_types):
            ssi.hparams[k].string_value = v

            if k in hparam_domain_discrete:
                domain_discrete = struct_pb2.ListValue(
                    values=[
                        struct_pb2.Value(string_value=d)
                        for d in hparam_domain_discrete[k]
                    ]
                )
            else:
                domain_discrete = None

            hps.append(
                HParamInfo(
                    name=k,
                    type=DataType.Value("DATA_TYPE_STRING"),
                    domain_discrete=domain_discrete,
                )
            )
            continue

        if isinstance(v, bool):
            ssi.hparams[k].bool_value = v

            if k in hparam_domain_discrete:
                domain_discrete = struct_pb2.ListValue(
                    values=[
                        struct_pb2.Value(bool_value=d)
                        for d in hparam_domain_discrete[k]
                    ]
                )
            else:
                domain_discrete = None

            hps.append(
                HParamInfo(
                    name=k,
                    type=DataType.Value("DATA_TYPE_BOOL"),
                    domain_discrete=domain_discrete,
                )
            )
            continue

        if isinstance(v, torch.Tensor):
            v = make_np(v)[0]
            ssi.hparams[k].number_value = v
            hps.append(HParamInfo(name=k, type=DataType.Value("DATA_TYPE_FLOAT64")))
            continue
        raise ValueError(
            "value should be one of int, float, str, bool, or torch.Tensor"
        )

    content = HParamsPluginData(session_start_info=ssi, version=PLUGIN_DATA_VERSION)
    smd = SummaryMetadata(
        plugin_data=SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=content.SerializeToString()
        )
    )
    ssi = Summary(value=[Summary.Value(tag=SESSION_START_INFO_TAG, metadata=smd)])

    mts = [MetricInfo(name=MetricName(tag=k)) for k in metric_dict.keys()]

    exp = Experiment(hparam_infos=hps, metric_infos=mts)

    content = HParamsPluginData(experiment=exp, version=PLUGIN_DATA_VERSION)
    smd = SummaryMetadata(
        plugin_data=SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=content.SerializeToString()
        )
    )
    exp = Summary(value=[Summary.Value(tag=EXPERIMENT_TAG, metadata=smd)])

    sei = SessionEndInfo(status=Status.Value("STATUS_SUCCESS"))
    content = HParamsPluginData(session_end_info=sei, version=PLUGIN_DATA_VERSION)
    smd = SummaryMetadata(
        plugin_data=SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=content.SerializeToString()
        )
    )
    sei = Summary(value=[Summary.Value(tag=SESSION_END_INFO_TAG, metadata=smd)])

    return exp, ssi, sei


def scalar(name, tensor, collections=None, new_style=False, double_precision=False):
    """Outputs a `Summary` protocol buffer containing a single scalar value.
    The generated Summary has a Tensor.proto containing the input Tensor.
    Args:
      name: A name for the generated node. Will also serve as the series name in
        TensorBoard.
      tensor: A real numeric Tensor containing a single value.
      collections: Optional list of graph collections keys. The new summary op is
        added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
      new_style: Whether to use new style (tensor field) or old style (simple_value
        field). New style could lead to faster data loading.
    Returns:
      A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.
    Raises:
      ValueError: If tensor has the wrong shape or type.
    """
    tensor = make_np(tensor).squeeze()
    assert (
        tensor.ndim == 0
    ), f"Tensor should contain one element (0 dimensions). Was given size: {tensor.size} and {tensor.ndim} dimensions."
    # python float is double precision in numpy
    scalar = float(tensor)
    if new_style:
        tensor_proto = TensorProto(float_val=[scalar], dtype="DT_FLOAT")
        if double_precision:
            tensor_proto = TensorProto(double_val=[scalar], dtype="DT_DOUBLE")

        plugin_data = SummaryMetadata.PluginData(plugin_name="scalars")
        smd = SummaryMetadata(plugin_data=plugin_data)
        return Summary(
            value=[
                Summary.Value(
                    tag=name,
                    tensor=tensor_proto,
                    metadata=smd,
                )
            ]
        )
    else:
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
    hist = HistogramProto(
        min=min,
        max=max,
        num=num,
        sum=sum,
        sum_squares=sum_squares,
        bucket_limit=bucket_limits,
        bucket=bucket_counts,
    )
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
    values = make_np(values)
    hist = make_histogram(values.astype(float), bins, max_bins)
    return Summary(value=[Summary.Value(tag=name, histo=hist)])


def make_histogram(values, bins, max_bins=None):
    """Convert values into a histogram proto using logic from histogram.cc."""
    if values.size == 0:
        raise ValueError("The input has no element.")
    values = values.reshape(-1)
    counts, limits = np.histogram(values, bins=bins)
    num_bins = len(counts)
    if max_bins is not None and num_bins > max_bins:
        subsampling = num_bins // max_bins
        subsampling_remainder = num_bins % subsampling
        if subsampling_remainder != 0:
            counts = np.pad(
                counts,
                pad_width=[[0, subsampling - subsampling_remainder]],
                mode="constant",
                constant_values=0,
            )
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
    counts = (
        counts[start - 1 : end] if start > 0 else np.concatenate([[0], counts[:end]])
    )
    limits = limits[start : end + 1]

    if counts.size == 0 or limits.size == 0:
        raise ValueError("The histogram is empty, please file a bug report.")

    sum_sq = values.dot(values)
    return HistogramProto(
        min=values.min(),
        max=values.max(),
        num=len(values),
        sum=values.sum(),
        sum_squares=sum_sq,
        bucket_limit=limits.tolist(),
        bucket=counts.tolist(),
    )


def image(tag, tensor, rescale=1, dataformats="NCHW"):
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
    tensor = make_np(tensor)
    tensor = convert_to_HWC(tensor, dataformats)
    # Do not assume that user passes in values in [0, 255], use data type to detect
    scale_factor = _calc_scale_factor(tensor)
    tensor = tensor.astype(np.float32)
    tensor = (tensor * scale_factor).astype(np.uint8)
    image = make_image(tensor, rescale=rescale)
    return Summary(value=[Summary.Value(tag=tag, image=image)])


def image_boxes(
    tag, tensor_image, tensor_boxes, rescale=1, dataformats="CHW", labels=None
):
    """Outputs a `Summary` protocol buffer with images."""
    tensor_image = make_np(tensor_image)
    tensor_image = convert_to_HWC(tensor_image, dataformats)
    tensor_boxes = make_np(tensor_boxes)
    tensor_image = tensor_image.astype(np.float32) * _calc_scale_factor(tensor_image)
    image = make_image(
        tensor_image.astype(np.uint8), rescale=rescale, rois=tensor_boxes, labels=labels
    )
    return Summary(value=[Summary.Value(tag=tag, image=image)])


def draw_boxes(disp_image, boxes, labels=None):
    # xyxy format
    num_boxes = boxes.shape[0]
    list_gt = range(num_boxes)
    for i in list_gt:
        disp_image = _draw_single_box(
            disp_image,
            boxes[i, 0],
            boxes[i, 1],
            boxes[i, 2],
            boxes[i, 3],
            display_str=None if labels is None else labels[i],
            color="Red",
        )
    return disp_image


def make_image(tensor, rescale=1, rois=None, labels=None):
    """Convert a numpy representation of an image to Image protobuf"""
    from PIL import Image

    height, width, channel = tensor.shape
    scaled_height = int(height * rescale)
    scaled_width = int(width * rescale)
    image = Image.fromarray(tensor)
    if rois is not None:
        image = draw_boxes(image, rois, labels=labels)
    try:
        ANTIALIAS = Image.Resampling.LANCZOS
    except AttributeError:
        ANTIALIAS = Image.ANTIALIAS
    image = image.resize((scaled_width, scaled_height), ANTIALIAS)
    import io
    output = io.BytesIO()
    image.save(output, format="PNG")
    image_string = output.getvalue()
    output.close()
    return Summary.Image(
        height=height,
        width=width,
        colorspace=channel,
        encoded_image_string=image_string,
    )


def video(tag, tensor, fps=4):
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
        print("add_video needs package moviepy")
        return
    try:
        from moviepy import editor as mpy
    except ImportError:
        print(
            "moviepy is installed, but can't import moviepy.editor.",
            "Some packages could be missing [imageio, requests]",
        )
        return
    import tempfile

    t, h, w, c = tensor.shape

    # encode sequence of images into gif string
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

    filename = tempfile.NamedTemporaryFile(suffix=".gif", delete=False).name
    try:  # newer version of moviepy use logger instead of progress_bar argument.
        clip.write_gif(filename, verbose=False, logger=None)
    except TypeError:
        try:  # older version of moviepy does not support progress_bar argument.
            clip.write_gif(filename, verbose=False, progress_bar=False)
        except TypeError:
            clip.write_gif(filename, verbose=False)

    with open(filename, "rb") as f:
        tensor_string = f.read()

    try:
        os.remove(filename)
    except OSError:
        logger.warning("The temporary file used by moviepy cannot be deleted.")

    return Summary.Image(
        height=h, width=w, colorspace=c, encoded_image_string=tensor_string
    )


def audio(tag, tensor, sample_rate=44100):
    array = make_np(tensor)
    array = array.squeeze()
    if abs(array).max() > 1:
        print("warning: audio amplitude out of range, auto clipped.")
        array = array.clip(-1, 1)
    assert array.ndim == 1, "input tensor should be 1 dimensional."
    array = (array * np.iinfo(np.int16).max).astype("<i2")

    import io
    import wave

    fio = io.BytesIO()
    with wave.open(fio, "wb") as wave_write:
        wave_write.setnchannels(1)
        wave_write.setsampwidth(2)
        wave_write.setframerate(sample_rate)
        wave_write.writeframes(array.data)
    audio_string = fio.getvalue()
    fio.close()
    audio = Summary.Audio(
        sample_rate=sample_rate,
        num_channels=1,
        length_frames=array.shape[-1],
        encoded_audio_string=audio_string,
        content_type="audio/wav",
    )
    return Summary(value=[Summary.Value(tag=tag, audio=audio)])


def custom_scalars(layout):
    categories = []
    for k, v in layout.items():
        charts = []
        for chart_name, chart_meatadata in v.items():
            tags = chart_meatadata[1]
            if chart_meatadata[0] == "Margin":
                assert len(tags) == 3
                mgcc = layout_pb2.MarginChartContent(
                    series=[
                        layout_pb2.MarginChartContent.Series(
                            value=tags[0], lower=tags[1], upper=tags[2]
                        )
                    ]
                )
                chart = layout_pb2.Chart(title=chart_name, margin=mgcc)
            else:
                mlcc = layout_pb2.MultilineChartContent(tag=tags)
                chart = layout_pb2.Chart(title=chart_name, multiline=mlcc)
            charts.append(chart)
        categories.append(layout_pb2.Category(title=k, chart=charts))

    layout = layout_pb2.Layout(category=categories)
    plugin_data = SummaryMetadata.PluginData(plugin_name="custom_scalars")
    smd = SummaryMetadata(plugin_data=plugin_data)
    tensor = TensorProto(
        dtype="DT_STRING",
        string_val=[layout.SerializeToString()],
        tensor_shape=TensorShapeProto(),
    )
    return Summary(
        value=[
            Summary.Value(tag="custom_scalars__config__", tensor=tensor, metadata=smd)
        ]
    )


def text(tag, text):
    plugin_data = SummaryMetadata.PluginData(
        plugin_name="text", content=TextPluginData(version=0).SerializeToString()
    )
    smd = SummaryMetadata(plugin_data=plugin_data)
    tensor = TensorProto(
        dtype="DT_STRING",
        string_val=[text.encode(encoding="utf_8")],
        tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),
    )
    return Summary(
        value=[Summary.Value(tag=tag + "/text_summary", metadata=smd, tensor=tensor)]
    )


def pr_curve_raw(
    tag, tp, fp, tn, fn, precision, recall, num_thresholds=127, weights=None
):
    if num_thresholds > 127:  # weird, value > 127 breaks protobuf
        num_thresholds = 127
    data = np.stack((tp, fp, tn, fn, precision, recall))
    pr_curve_plugin_data = PrCurvePluginData(
        version=0, num_thresholds=num_thresholds
    ).SerializeToString()
    plugin_data = SummaryMetadata.PluginData(
        plugin_name="pr_curves", content=pr_curve_plugin_data
    )
    smd = SummaryMetadata(plugin_data=plugin_data)
    tensor = TensorProto(
        dtype="DT_FLOAT",
        float_val=data.reshape(-1).tolist(),
        tensor_shape=TensorShapeProto(
            dim=[
                TensorShapeProto.Dim(size=data.shape[0]),
                TensorShapeProto.Dim(size=data.shape[1]),
            ]
        ),
    )
    return Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor)])


def pr_curve(tag, labels, predictions, num_thresholds=127, weights=None):
    # weird, value > 127 breaks protobuf
    num_thresholds = min(num_thresholds, 127)
    data = compute_curve(
        labels, predictions, num_thresholds=num_thresholds, weights=weights
    )
    pr_curve_plugin_data = PrCurvePluginData(
        version=0, num_thresholds=num_thresholds
    ).SerializeToString()
    plugin_data = SummaryMetadata.PluginData(
        plugin_name="pr_curves", content=pr_curve_plugin_data
    )
    smd = SummaryMetadata(plugin_data=plugin_data)
    tensor = TensorProto(
        dtype="DT_FLOAT",
        float_val=data.reshape(-1).tolist(),
        tensor_shape=TensorShapeProto(
            dim=[
                TensorShapeProto.Dim(size=data.shape[0]),
                TensorShapeProto.Dim(size=data.shape[1]),
            ]
        ),
    )
    return Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor)])


# https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/summary.py
def compute_curve(labels, predictions, num_thresholds=None, weights=None):
    _MINIMUM_COUNT = 1e-7

    if weights is None:
        weights = 1.0

    # Compute bins of true positives and false positives.
    bucket_indices = np.int32(np.floor(predictions * (num_thresholds - 1)))
    float_labels = labels.astype(np.float64)
    histogram_range = (0, num_thresholds - 1)
    tp_buckets, _ = np.histogram(
        bucket_indices,
        bins=num_thresholds,
        range=histogram_range,
        weights=float_labels * weights,
    )
    fp_buckets, _ = np.histogram(
        bucket_indices,
        bins=num_thresholds,
        range=histogram_range,
        weights=(1.0 - float_labels) * weights,
    )

    # Obtain the reverse cumulative sum.
    tp = np.cumsum(tp_buckets[::-1])[::-1]
    fp = np.cumsum(fp_buckets[::-1])[::-1]
    tn = fp[0] - fp
    fn = tp[0] - tp
    precision = tp / np.maximum(_MINIMUM_COUNT, tp + fp)
    recall = tp / np.maximum(_MINIMUM_COUNT, tp + fn)
    return np.stack((tp, fp, tn, fn, precision, recall))


def _get_tensor_summary(
    name, display_name, description, tensor, content_type, components, json_config
):
    """Creates a tensor summary with summary metadata.

    Args:
      name: Uniquely identifiable name of the summary op. Could be replaced by
        combination of name and type to make it unique even outside of this
        summary.
      display_name: Will be used as the display name in TensorBoard.
        Defaults to `name`.
      description: A longform readable description of the summary data. Markdown
        is supported.
      tensor: Tensor to display in summary.
      content_type: Type of content inside the Tensor.
      components: Bitmask representing present parts (vertices, colors, etc.) that
        belong to the summary.
      json_config: A string, JSON-serialized dictionary of ThreeJS classes
        configuration.

    Returns:
      Tensor summary with metadata.
    """
    import torch
    from tensorboard.plugins.mesh import metadata

    tensor = torch.as_tensor(tensor)

    tensor_metadata = metadata.create_summary_metadata(
        name,
        display_name,
        content_type,
        components,
        tensor.shape,
        description,
        json_config=json_config,
    )

    tensor = TensorProto(
        dtype="DT_FLOAT",
        float_val=tensor.reshape(-1).tolist(),
        tensor_shape=TensorShapeProto(
            dim=[
                TensorShapeProto.Dim(size=tensor.shape[0]),
                TensorShapeProto.Dim(size=tensor.shape[1]),
                TensorShapeProto.Dim(size=tensor.shape[2]),
            ]
        ),
    )

    tensor_summary = Summary.Value(
        tag=metadata.get_instance_name(name, content_type),
        tensor=tensor,
        metadata=tensor_metadata,
    )

    return tensor_summary


def _get_json_config(config_dict):
    """Parses and returns JSON string from python dictionary."""
    json_config = "{}"
    if config_dict is not None:
        json_config = json.dumps(config_dict, sort_keys=True)
    return json_config


# https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/mesh/summary.py
def mesh(
    tag, vertices, colors, faces, config_dict, display_name=None, description=None
):
    """Outputs a merged `Summary` protocol buffer with a mesh/point cloud.

    Args:
      tag: A name for this summary operation.
      vertices: Tensor of shape `[dim_1, ..., dim_n, 3]` representing the 3D
        coordinates of vertices.
      faces: Tensor of shape `[dim_1, ..., dim_n, 3]` containing indices of
        vertices within each triangle.
      colors: Tensor of shape `[dim_1, ..., dim_n, 3]` containing colors for each
        vertex.
      display_name: If set, will be used as the display name in TensorBoard.
        Defaults to `name`.
      description: A longform readable description of the summary data. Markdown
        is supported.
      config_dict: Dictionary with ThreeJS classes names and configuration.

    Returns:
      Merged summary for mesh/point cloud representation.
    """
    from tensorboard.plugins.mesh.plugin_data_pb2 import MeshPluginData
    from tensorboard.plugins.mesh import metadata

    json_config = _get_json_config(config_dict)

    summaries = []
    tensors = [
        (vertices, MeshPluginData.VERTEX),
        (faces, MeshPluginData.FACE),
        (colors, MeshPluginData.COLOR),
    ]
    tensors = [tensor for tensor in tensors if tensor[0] is not None]
    components = metadata.get_components_bitmask(
        [content_type for (tensor, content_type) in tensors]
    )

    for tensor, content_type in tensors:
        summaries.append(
            _get_tensor_summary(
                tag,
                display_name,
                description,
                tensor,
                content_type,
                components,
                json_config,
            )
        )

    return Summary(value=summaries)
