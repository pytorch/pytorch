from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from caffe2.proto import caffe2_pb2
import caffe2.python.cnn as cnn
import caffe2.python.core as core
import caffe2.contrib.tensorboard.tensorboard_exporter as tb

EXPECTED = """
node {
  name: "conv1/XavierFill"
  op: "XavierFill"
  device: "/gpu:0"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 96
          }
          dim {
            size: 3
          }
          dim {
            size: 11
          }
          dim {
            size: 11
          }
        }
      }
    }
  }
}
node {
  name: "conv1/ConstantFill"
  op: "ConstantFill"
  device: "/gpu:0"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 96
          }
        }
      }
    }
  }
}
node {
  name: "classifier/XavierFill"
  op: "XavierFill"
  device: "/gpu:0"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 1000
          }
          dim {
            size: 4096
          }
        }
      }
    }
  }
}
node {
  name: "classifier/ConstantFill"
  op: "ConstantFill"
  device: "/gpu:0"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 1000
          }
        }
      }
    }
  }
}
node {
  name: "ImageInput"
  op: "ImageInput"
  input: "db"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "is_test"
    value {
      i: 0
    }
  }
  attr {
    key: "use_cudnn"
    value {
      i: 1
    }
  }
}
node {
  name: "NHWC2NCHW"
  op: "NHWC2NCHW"
  input: "data_nhwc"
  device: "/gpu:0"
}
node {
  name: "conv1/Conv"
  op: "Conv"
  input: "data"
  input: "conv1/conv1_w"
  input: "conv1/conv1_b"
  device: "/gpu:0"
  attr {
    key: "exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "kernel"
    value {
      i: 11
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "stride"
    value {
      i: 4
    }
  }
}
node {
  name: "conv1/Relu"
  op: "Relu"
  input: "conv1/conv1"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
}
node {
  name: "conv1/MaxPool"
  op: "MaxPool"
  input: "conv1/conv1_1"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "kernel"
    value {
      i: 2
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "stride"
    value {
      i: 2
    }
  }
}
node {
  name: "classifier/FC"
  op: "FC"
  input: "conv1/pool1"
  input: "classifier/fc_w"
  input: "classifier/fc_b"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "use_cudnn"
    value {
      i: 1
    }
  }
}
node {
  name: "classifier/Softmax"
  op: "Softmax"
  input: "classifier/fc"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
}
node {
  name: "classifier/LabelCrossEntropy"
  op: "LabelCrossEntropy"
  input: "classifier/pred"
  input: "label"
  device: "/gpu:0"
}
node {
  name: "classifier/AveragedLoss"
  op: "AveragedLoss"
  input: "classifier/xent"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/ConstantFill"
  op: "ConstantFill"
  input: "classifier/loss"
  device: "/gpu:0"
  attr {
    key: "value"
    value {
      f: 1.0
    }
  }
}
node {
  name: "GRADIENTS/classifier/AveragedLossGradient"
  op: "AveragedLossGradient"
  input: "classifier/xent"
  input: "GRADIENTS/classifier/loss_autogen_grad"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/LabelCrossEntropyGradient"
  op: "LabelCrossEntropyGradient"
  input: "classifier/pred"
  input: "label"
  input: "GRADIENTS/classifier/xent_grad"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/SoftmaxGradient"
  op: "SoftmaxGradient"
  input: "classifier/pred"
  input: "GRADIENTS/classifier/pred_grad"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
}
node {
  name: "GRADIENTS/c/FCGradient"
  op: "FCGradient"
  input: "conv1/pool1"
  input: "classifier/fc_w"
  input: "GRADIENTS/classifier/fc_grad"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "use_cudnn"
    value {
      i: 1
    }
  }
}
node {
  name: "GRADIENTS/conv1/MaxPoolGradient"
  op: "MaxPoolGradient"
  input: "conv1/conv1_1"
  input: "conv1/pool1"
  input: "GRADIENTS/conv1/pool1_grad"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "kernel"
    value {
      i: 2
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "stride"
    value {
      i: 2
    }
  }
}
node {
  name: "GRADIENTS/conv1/ReluGradient"
  op: "ReluGradient"
  input: "conv1/conv1_1"
  input: "GRADIENTS/conv1/conv1_grad"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
}
node {
  name: "GRADIENTS/ConvGradient"
  op: "ConvGradient"
  input: "data"
  input: "conv1/conv1_w"
  input: "GRADIENTS/conv1/conv1_grad_1"
  device: "/gpu:0"
  attr {
    key: "exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "kernel"
    value {
      i: 11
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "stride"
    value {
      i: 4
    }
  }
}
node {
  name: "GRADIENTS/NCHW2NHWC"
  op: "NCHW2NHWC"
  input: "GRADIENTS/data_grad"
  device: "/gpu:0"
}
node {
  name: "conv1/conv1_w"
  op: "Blob"
  input: "conv1/XavierFill:0"
  device: "/gpu:0"
}
node {
  name: "classifier/fc"
  op: "Blob"
  input: "classifier/FC:0"
  device: "/gpu:0"
}
node {
  name: "data_nhwc"
  op: "Blob"
  input: "ImageInput:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/conv1/conv1_b_grad"
  op: "Blob"
  input: "GRADIENTS/ConvGradient:1"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/pred_grad"
  op: "Blob"
  input: "GRADIENTS/classifier/LabelCrossEntropyGradient:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/fc_grad"
  op: "Blob"
  input: "GRADIENTS/classifier/SoftmaxGradient:0"
  device: "/gpu:0"
}
node {
  name: "conv1/conv1_b"
  op: "Blob"
  input: "conv1/ConstantFill:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/fc_b_grad"
  op: "Blob"
  input: "GRADIENTS/c/FCGradient:1"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/fc_w_grad"
  op: "Blob"
  input: "GRADIENTS/c/FCGradient:0"
  device: "/gpu:0"
}
node {
  name: "label"
  op: "Blob"
  input: "ImageInput:1"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/data_grad"
  op: "Blob"
  input: "GRADIENTS/ConvGradient:2"
  device: "/gpu:0"
}
node {
  name: "classifier/loss"
  op: "Blob"
  input: "classifier/AveragedLoss:0"
  device: "/gpu:0"
}
node {
  name: "conv1/conv1"
  op: "Blob"
  input: "conv1/Conv:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/conv1/conv1_grad"
  op: "Blob"
  input: "GRADIENTS/conv1/MaxPoolGradient:0"
  device: "/gpu:0"
}
node {
  name: "classifier/xent"
  op: "Blob"
  input: "classifier/LabelCrossEntropy:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/loss_autogen_grad"
  op: "Blob"
  input: "GRADIENTS/classifier/ConstantFill:0"
  device: "/gpu:0"
}
node {
  name: "classifier/fc_w"
  op: "Blob"
  input: "classifier/XavierFill:0"
  device: "/gpu:0"
}
node {
  name: "conv1/conv1_1"
  op: "Blob"
  input: "conv1/Relu:0"
  device: "/gpu:0"
}
node {
  name: "db"
  op: "Placeholder"
}
node {
  name: "classifier/pred"
  op: "Blob"
  input: "classifier/Softmax:0"
  device: "/gpu:0"
}
node {
  name: "classifier/fc_b"
  op: "Blob"
  input: "classifier/ConstantFill:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/xent_grad"
  op: "Blob"
  input: "GRADIENTS/classifier/AveragedLossGradient:0"
  device: "/gpu:0"
}
node {
  name: "data"
  op: "Blob"
  input: "NHWC2NCHW:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/conv1/conv1_w_grad"
  op: "Blob"
  input: "GRADIENTS/ConvGradient:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/conv1/conv1_grad_1"
  op: "Blob"
  input: "GRADIENTS/conv1/ReluGradient:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/data_nhwc_grad"
  op: "Blob"
  input: "GRADIENTS/NCHW2NHWC:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/conv1/pool1_grad"
  op: "Blob"
  input: "GRADIENTS/c/FCGradient:2"
  device: "/gpu:0"
}
node {
  name: "conv1/pool1"
  op: "Blob"
  input: "conv1/MaxPool:0"
  device: "/gpu:0"
}
"""


class TensorboardExporterTest(unittest.TestCase):
    def test_that_operators_gets_non_colliding_names(self):
        op = caffe2_pb2.OperatorDef()
        op.type = 'foo'
        op.input.extend(['foo'])
        tb._fill_missing_operator_names([op])
        self.assertEqual(op.input[0], 'foo')
        self.assertEqual(op.name, 'foo_1')

    def test_that_replacing_colons_gives_non_colliding_names(self):
        # .. and update shapes
        op = caffe2_pb2.OperatorDef()
        op.name = 'foo:0'
        op.input.extend(['foo:0', 'foo$0'])
        shapes = {'foo:0': [1]}
        track_blob_names = tb._get_blob_names([op])
        tb._replace_colons(shapes, track_blob_names, [op], '$')
        self.assertEqual(op.input[0], 'foo$0')
        self.assertEqual(op.input[1], 'foo$0_1')
        # Collision but blobs and op names are handled later by
        # _fill_missing_operator_names.
        self.assertEqual(op.name, 'foo$0')
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes['foo$0'], [1])
        self.assertEqual(len(track_blob_names), 2)
        self.assertEqual(track_blob_names['foo$0'], 'foo:0')
        self.assertEqual(track_blob_names['foo$0_1'], 'foo$0')

    def test_that_adding_gradient_scope_does_no_fancy_renaming(self):
        # because it cannot create collisions
        op = caffe2_pb2.OperatorDef()
        op.name = 'foo_grad'
        op.input.extend(['foo_grad', 'foo_grad_1'])
        shapes = {'foo_grad': [1]}
        track_blob_names = tb._get_blob_names([op])
        tb._add_gradient_scope(shapes, track_blob_names, [op])
        self.assertEqual(op.input[0], 'GRADIENTS/foo_grad')
        self.assertEqual(op.input[1], 'GRADIENTS/foo_grad_1')
        self.assertEqual(op.name, 'GRADIENTS/foo_grad')
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes['GRADIENTS/foo_grad'], [1])
        self.assertEqual(len(track_blob_names), 2)
        self.assertEqual(
            track_blob_names['GRADIENTS/foo_grad'], 'foo_grad')
        self.assertEqual(
            track_blob_names['GRADIENTS/foo_grad_1'], 'foo_grad_1')

    def test_that_auto_ssa_gives_non_colliding_names(self):
        op1 = caffe2_pb2.OperatorDef()
        op1.output.extend(['foo'])
        op2 = caffe2_pb2.OperatorDef()
        op2.input.extend(['foo'])
        op2.output.extend(['foo'])
        op2.output.extend(['foo_1'])
        shapes = {'foo': [1], 'foo_1': [2]}
        track_blob_names = tb._get_blob_names([op1, op2])
        tb._convert_to_ssa(shapes, track_blob_names, [op1, op2])
        self.assertEqual(op1.output[0], 'foo')
        self.assertEqual(op2.input[0], 'foo')
        self.assertEqual(op2.output[0], 'foo_1')
        # Unfortunate name but we do not parse original `_` for now.
        self.assertEqual(op2.output[1], 'foo_1_1')
        self.assertEqual(len(shapes), 3)
        self.assertEqual(shapes['foo'], [1])
        self.assertEqual(shapes['foo_1'], [1])
        self.assertEqual(shapes['foo_1_1'], [2])
        self.assertEqual(len(track_blob_names), 3)
        self.assertEqual(track_blob_names['foo'], 'foo')
        self.assertEqual(track_blob_names['foo_1'], 'foo')
        self.assertEqual(track_blob_names['foo_1_1'], 'foo_1')

    def test_simple_cnnmodel(self):
        model = cnn.CNNModelHelper("NCHW", name="overfeat")
        data, label = model.ImageInput(["db"], ["data", "label"], is_test=0)
        with core.NameScope("conv1"):
            conv1 = model.Conv(data, "conv1", 3, 96, 11, stride=4)
            relu1 = model.Relu(conv1, conv1)
            pool1 = model.MaxPool(relu1, "pool1", kernel=2, stride=2)
        with core.NameScope("classifier"):
            fc = model.FC(pool1, "fc", 4096, 1000)
            pred = model.Softmax(fc, "pred")
            xent = model.LabelCrossEntropy([pred, label], "xent")
            loss = model.AveragedLoss(xent, "loss")
        model.net.RunAllOnGPU()
        model.param_init_net.RunAllOnGPU()
        model.AddGradientOperators([loss], skip=1)
        track_blob_names = {}
        graph = tb.cnn_to_graph_def(
            model,
            track_blob_names=track_blob_names,
            shapes={},
        )
        self.assertEqual(
            track_blob_names['GRADIENTS/conv1/conv1_b_grad'],
            'conv1/conv1_b_grad',
        )
        self.maxDiff = None
        # We can't guarantee the order in which they appear, so we sort
        # both before we compare them
        sep = "node {"
        expected = "\n".join(sorted(
            sep + "\n  " + part.strip()
            for part in EXPECTED.strip().split(sep)
            if part.strip()
        ))
        actual = "\n".join(sorted(
            sep + "\n  " + part.strip()
            for part in str(graph).strip().split(sep)
            if part.strip()
        ))
        self.assertMultiLineEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
