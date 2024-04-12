# @package onnx
# Module caffe2.python.onnx.backend_rep





from caffe2.python import core
from caffe2.proto import caffe2_pb2
from onnx.backend.base import BackendRep, namedtupledict

class Caffe2Rep(BackendRep):
    def __init__(self, init_net, predict_net, workspace, uninitialized):
        super().__init__()
        self.init_net = init_net
        self.predict_net = predict_net
        self.workspace = workspace
        # The list of uninitialized external_inputs in workspace, we need this to
        # pair the name with given sequence inputs.
        self.uninitialized = uninitialized
        self.nets_created = False
        self.ran_init_net = False

    @property
    def _name_scope(self):
        if self.predict_net.device_option.device_type == caffe2_pb2.CUDA:
            return 'gpu_{}'.format(self.predict_net.device_option.device_id)
        return ''

    def run(self, inputs, **kwargs):
        super().run(inputs, **kwargs)
        with core.DeviceScope(self.predict_net.device_option):
            if isinstance(inputs, dict):
                with core.NameScope(self._name_scope):
                    for key, value in inputs.items():
                        self.workspace.FeedBlob(key, value)
            elif isinstance(inputs, list) or isinstance(inputs, tuple):
                if len(self.uninitialized) != len(inputs):
                    raise RuntimeError('Expected {} values for uninitialized '
                                       'graph inputs ({}), but got {}.'.format(
                                           len(self.uninitialized),
                                           ', '.join(self.uninitialized),
                                           len(inputs)))
                for i, value in enumerate(inputs):
                    # namescope already baked into protobuf
                    self.workspace.FeedBlob(self.uninitialized[i], value)
            else:
                # single input
                self.workspace.FeedBlob(self.uninitialized[0], inputs)
            if not self.nets_created:
                self.workspace.CreateNet(self.init_net)
                self.workspace.CreateNet(self.predict_net)
                self.nets_created = True
            if not self.ran_init_net:
                self.workspace.RunNet(self.init_net.name)
                self.ran_init_net = True
            self.workspace.RunNet(self.predict_net.name)
        output_values = []
        for name in self.predict_net.external_output:
            try:
                output_values.append(self.workspace.FetchBlob(name))
            except Exception:
                output_values.append(self.workspace.FetchInt8Blob(name))
        return namedtupledict('Outputs',
                              self.predict_net.external_output)(*output_values)
