from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import click.testing
import numpy as np
import os
import tempfile
import unittest

from caffe2.python import brew, core, model_helper
import caffe2.contrib.tensorboard.tensorboard as tb
import caffe2.contrib.tensorboard.tensorboard_exporter as tb_exporter
import tensorflow as tf


class TensorboardTest(unittest.TestCase):

    def test_events(self):
        runner = click.testing.CliRunner()
        c2_dir = tempfile.mkdtemp()
        np.random.seed(1701)
        n_iters = 2
        blobs = ["w", "b"]
        data = np.random.randn(len(blobs), n_iters, 10)
        for i, blob in enumerate(blobs):
            with open(os.path.join(c2_dir, blob), "w") as f:
                for row in data[i]:
                    stats = [row.min(), row.max(), row.mean(), row.std()]
                    f.write(" ".join(str(s) for s in stats) + "\n")

        # Test error handling path
        with open(os.path.join(c2_dir, "not-a-summary"), "w") as f:
            f.write("not-a-summary")

        tf_dir = tempfile.mkdtemp()
        result = runner.invoke(
            tb.cli,
            ["tensorboard-events", "--c2-dir", c2_dir, "--tf-dir", tf_dir])
        self.assertEqual(result.exit_code, 0)
        entries = list(os.walk(tf_dir))
        self.assertEqual(len(entries), 1)
        ((d, _, (fname,)),) = entries
        self.assertEqual(tf_dir, d)
        events = list(tf.train.summary_iterator(os.path.join(tf_dir, fname)))
        self.assertEqual(len(events), n_iters + 1)
        events = events[1:]
        self.maxDiff = None
        self.assertEqual(len(events), 2)

    def test_tensorboard_graphs(self):
        model = model_helper.ModelHelper(name="overfeat")
        data, label = brew.image_input(
            model, ["db"], ["data", "label"], is_test=0
        )
        with core.NameScope("conv1"):
            conv1 = brew.conv(model, data, "conv1", 3, 96, 11, stride=4)
            relu1 = brew.relu(model, conv1, conv1)
            pool1 = brew.max_pool(model, relu1, "pool1", kernel=2, stride=2)
        with core.NameScope("classifier"):
            fc = brew.fc(model, pool1, "fc", 4096, 1000)
            pred = brew.softmax(model, fc, "pred")
            xent = model.LabelCrossEntropy([pred, label], "xent")
            loss = model.AveragedLoss(xent, "loss")
        model.AddGradientOperators([loss], skip=1)

        c2_dir = tempfile.mkdtemp()
        tf_dir = tempfile.mkdtemp()

        with open(os.path.join(c2_dir, "init"), "w") as f:
            f.write(str(model.param_init_net.Proto()))
        with open(os.path.join(c2_dir, "net"), "w") as f:
            f.write(str(model.net.Proto()))
        runner = click.testing.CliRunner()
        result = runner.invoke(
            tb.cli,
            ["tensorboard-graphs",
             "--c2-netdef", os.path.join(c2_dir, "init"),
             "--c2-netdef", os.path.join(c2_dir, "net"),
             "--tf-dir", tf_dir])
        self.assertEqual(result.exit_code, 0)
        entries = list(os.walk(tf_dir))
        self.assertEqual(len(entries), 1)
        ((d, _, (fname,)),) = entries
        self.assertEqual(tf_dir, d)
        events = list(tf.train.summary_iterator(os.path.join(tf_dir, fname)))
        self.assertEqual(len(events), 3)
        events = events[1:]
        nets = [model.param_init_net, model.net]
        for i, (event, net) in enumerate(zip(events, nets), start=1):
            self.assertEqual(event.step, i)
            self.assertEqual(event.wall_time, i)
            g = tf.GraphDef()
            g.ParseFromString(event.graph_def)
            self.assertMultiLineEqual(
                str(g),
                str(tb_exporter.nets_to_graph_def([net])))


if __name__ == "__main__":
    unittest.main()
