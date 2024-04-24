




import click
import collections
import logging
import numpy as np
import os

from caffe2.proto import caffe2_pb2
from caffe2.python import core
import caffe2.contrib.tensorboard.tensorboard_exporter as tb_exporter

try:
    # tensorboard>=1.14.0
    from tensorboard.compat.proto.summary_pb2 import Summary, HistogramProto
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.summary.writer.event_file_writer import EventFileWriter as FileWriter
except ImportError:
    from tensorflow.core.framework.summary_pb2 import Summary, HistogramProto
    from tensorflow.core.util.event_pb2 import Event
    try:
        # tensorflow>=1.0.0
        from tensorflow.summary import FileWriter
    except ImportError:
        # tensorflow<=0.12.1
        from tensorflow.train import SummaryWriter as FileWriter

class Config:
    HEIGHT = 600
    ASPECT_RATIO = 1.6


CODE_TEMPLATE = """
<script>
  function load() {{
    document.getElementById("{id}").pbtxt = {data};
  }}
</script>
<link rel="import"
  href="https://tensorboard.appspot.com/tf-graph-basic.build.html"
  onload=load()
>
<div style="height:{height}px">
  <tf-graph-basic id="{id}"></tf-graph-basic>
</div>
"""

IFRAME_TEMPLATE = """
<iframe
  seamless
  style="width:{width}px;height:{height}px;border:0"
  srcdoc="{code}">
</iframe>
"""


def _show_graph(graph_def):
    import IPython.display

    code = CODE_TEMPLATE.format(
        data=repr(str(graph_def)),
        id='graph' + str(np.random.rand()),
        height=Config.HEIGHT)

    iframe = IFRAME_TEMPLATE.format(
        code=code.replace('"', '&quot;'),
        width=Config.HEIGHT * Config.ASPECT_RATIO,
        height=Config.HEIGHT + 20)

    IPython.display.display(IPython.display.HTML(iframe))


def visualize_cnn(cnn, **kwargs):
    g = tb_exporter.cnn_to_graph_def(cnn, **kwargs)
    _show_graph(g)


def visualize_net(nets, **kwargs):
    g = tb_exporter.nets_to_graph_def(nets, **kwargs)
    _show_graph(g)


def visualize_ops(ops, **kwargs):
    g = tb_exporter.ops_to_graph_def(ops, **kwargs)
    _show_graph(g)


@click.group()
def cli():
    pass


def write_events(tf_dir, events):
    writer = FileWriter(tf_dir, len(events))
    for event in events:
        writer.add_event(event)
    writer.flush()
    writer.close()


def graph_def_to_event(step, graph_def):
    return Event(
        wall_time=step, step=step, graph_def=graph_def.SerializeToString())


@cli.command("tensorboard-graphs")  # type: ignore[arg-type, attr-defined]
@click.option("--c2-netdef", type=click.Path(exists=True, dir_okay=False),
              multiple=True)
@click.option("--tf-dir", type=click.Path(exists=True))
def tensorboard_graphs(c2_netdef, tf_dir):
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    def parse_net_def(path):
        import google.protobuf.text_format  # type: ignore[import]
        net_def = caffe2_pb2.NetDef()
        with open(path) as f:
            google.protobuf.text_format.Merge(f.read(), net_def)
        return core.Net(net_def)

    graph_defs = [tb_exporter.nets_to_graph_def([parse_net_def(path)])
                  for path in c2_netdef]
    events = [graph_def_to_event(i, graph_def)
              for (i, graph_def) in enumerate(graph_defs, start=1)]
    write_events(tf_dir, events)
    log.info("Wrote %s graphs to logdir %s", len(events), tf_dir)


@cli.command("tensorboard-events")  # type: ignore[arg-type, attr-defined]
@click.option("--c2-dir", type=click.Path(exists=True, file_okay=False),
              help="Root directory of the Caffe2 run")
@click.option("--tf-dir", type=click.Path(writable=True),
              help="Output path to the logdir used by TensorBoard")
def tensorboard_events(c2_dir, tf_dir):
    np.random.seed(1701)
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    S = collections.namedtuple('S', ['min', 'max', 'mean', 'std'])

    def parse_summary(filename):
        try:
            with open(filename) as f:
                rows = [(float(el) for el in line.split()) for line in f]
                return [S(*r) for r in rows]
        except Exception as e:
            log.exception(e)
            return None

    def get_named_summaries(root):
        summaries = [
            (fname, parse_summary(os.path.join(dirname, fname)))
            for dirname, _, fnames in os.walk(root)
            for fname in fnames
        ]
        return [(n, s) for (n, s) in summaries if s]

    def inferred_histo(summary, samples=1000):
        np.random.seed(
            hash(
                summary.std + summary.mean + summary.min + summary.max
            ) % np.iinfo(np.int32).max
        )
        samples = np.random.randn(samples) * summary.std + summary.mean
        samples = np.clip(samples, a_min=summary.min, a_max=summary.max)
        (hist, edges) = np.histogram(samples)
        upper_edges = edges[1:]
        r = HistogramProto(
            min=summary.min,
            max=summary.max,
            num=len(samples),
            sum=samples.sum(),
            sum_squares=(samples * samples).sum())
        r.bucket_limit.extend(upper_edges)
        r.bucket.extend(hist)
        return r

    def named_summaries_to_events(named_summaries):
        names = [n for (n, _) in named_summaries]
        summaries = [s for (_, s) in named_summaries]
        summaries = list(zip(*summaries))

        def event(step, values):
            s = Summary()
            scalar = [
                Summary.Value(
                    tag="{}/{}".format(name, field),
                    simple_value=v)
                for name, value in zip(names, values)
                for field, v in value._asdict().items()]
            hist = [
                Summary.Value(
                    tag="{}/inferred_normal_hist".format(name),
                    histo=inferred_histo(value))
                for name, value in zip(names, values)
            ]
            s.value.extend(scalar + hist)
            return Event(wall_time=int(step), step=step, summary=s)

        return [event(step, values)
                for step, values in enumerate(summaries, start=1)]

    named_summaries = get_named_summaries(c2_dir)
    events = named_summaries_to_events(named_summaries)
    write_events(tf_dir, events)
    log.info("Wrote %s events to logdir %s", len(events), tf_dir)


if __name__ == "__main__":
    cli()  # type: ignore[misc]
