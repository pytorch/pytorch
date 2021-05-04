import io
import torch.onnx
import onnx
from caffe2.python.onnx.backend import Caffe2Backend
from caffe2.python.core import BlobReference, Net


_next_idx = 0
# Clone net takes a dict instead of a lambda
# It should probably take a lambda, it is more flexible
# We fake dict here


class _FakeDict(object):
    def __init__(self, fn):
        self.fn = fn

    def get(self, name, _):
        return self.fn(name)


def PyTorchModule(helper, model, sample_arguments, caffe2_inputs, prefix_name=None):
    """
    Embed an ONNX-exportable PyTorch Model into a Caffe2 model being built.

    Args:
        helper (caffe2.python.core.ModelHelder): the model helper where
            this imported network should be inserted
        model (torch.nn.Module): the model to be exported
        sample_arguments (tuple of arguments): the inputs to
            the model, e.g., such that ``model(*args)`` is a valid
            invocation of the model.  Any non-Variable arguments will
            be hard-coded into the exported model; any Variable arguments
            will become inputs of the exported model, in the order they
            occur in args.  If args is a Variable, this is equivalent
            to having called it with a 1-ary tuple of that Variable.
            (Note: passing keyword arguments to the model is not currently
            supported.  Give us a shout if you need it.)
        caffe2_inputs (list of str or caffe2.python.core.BlobReference): the
           caffe2 Blobs that should be inputs to this network. Must be
           the same length as sample_arguments
        prefix_name: prefix name to add to each member of the blob, if None then
           a fresh prefix pytorch_input_N/ is used
    Returns:
        A tuple of caffe2.python.core.BlobReference objects referring to the
        models outputs, or a single BlobReference when the model returns a single
        value.
    """
    if prefix_name is None:
        global _next_idx
        prefix_name = 'pytorch_import_' + str(_next_idx) + '/'
        _next_idx += 1

    # TODO: handle the case where model cannot be exported
    # and embed as a Python op in Caffe2
    f = io.BytesIO()
    torch.onnx.export(
        model, sample_arguments, f, export_params=True)
    onnx_model = onnx.load(io.BytesIO(f.getvalue()))
    init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(
        onnx_model)

    initialized = set([x.name for x in onnx_model.graph.initializer])
    uninitialized_inputs = {x.name: i for i, x in enumerate(
        onnx_model.graph.input) if x.name not in initialized}

    if(len(uninitialized_inputs) != len(caffe2_inputs)):
        raise ValueError('Expected {} inputs but found {}'.format(
            len(uninitialized_inputs), len(caffe2_inputs)))

    def remap_blob_name(name):
        if name in uninitialized_inputs:
            idx = uninitialized_inputs[name]
            return str(caffe2_inputs[idx])
        return prefix_name + name

    predict_net = Net(predict_net).Clone('anon', _FakeDict(remap_blob_name))
    helper.net.AppendNet(predict_net)

    init_net = Net(init_net).Clone('anon', _FakeDict(remap_blob_name))
    helper.param_init_net.AppendNet(init_net)

    results = tuple([BlobReference(remap_blob_name(x.name), helper.net)
                     for x in onnx_model.graph.output])
    return results
