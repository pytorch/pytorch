from caffe2.python import workspace, core
import numpy as np

from utils import NUM_LOOP_ITERS

workspace.GlobalInit(['caffe2'])

def add_blob(ws, blob_name, tensor_size):
    blob_tensor = np.random.randn(*tensor_size).astype(np.float32)
    ws.FeedBlob(blob_name, blob_tensor)

class C2SimpleNet:
    """
    This module constructs a net with 'op_name' operator. The net consist
    a series of such operator.
    It initializes the workspace with input blob equal to the number of parameters
    needed for the op.
    Provides forward method to run the net niter times.
    """
    def __init__(self, op_name, num_inputs=1, debug=False):
        self.input_names = []
        self.net = core.Net("framework_benchmark_net")
        self.input_names = [f"in_{i}" for i in range(num_inputs)]
        for i in range(num_inputs):
            add_blob(workspace, self.input_names[i], [1])
        self.net.AddExternalInputs(self.input_names)
        op_constructor = getattr(self.net, op_name)
        op_constructor(self.input_names)
        self.output_name = self.net._net.op[-1].output
        print(f"Benchmarking op {op_name}:")
        for _ in range(NUM_LOOP_ITERS):
            output_name = self.net._net.op[-1].output
            self.input_names[-1] = output_name[0]
            assert len(self.input_names) == num_inputs
            op_constructor(self.input_names)
        workspace.CreateNet(self.net)
        if debug:
            print(self.net._net)

    def forward(self, niters):
        workspace.RunNet(self.net, niters, False)
