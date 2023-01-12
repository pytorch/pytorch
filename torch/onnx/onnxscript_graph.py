class OSNode(object):
    def __init__(self, op_name, input_names, output_names, scope):
        """Create Node.
        
        %26 : Long(device=cpu) = prim::Constant[value={5}]()
        %24 : Float(2, 3, 4, strides=[12, 4, 1], requires_grad=0, device=cpu) = aten::add(%16, %26, %25), scope: __main__.MixedOpsModel::

        """
        self._op = op_name
        self._op_type = op_name.split("::")[0]
        self._input = list(input_names)
        self._output = list(output_names)
        self._scope = scope

        print("===== OS Node is constructed. =====")

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output


class OSGraph(object):
    def __init__(self):
        self._nodes = {}
        print("===== OS Graph is constructed. =====")

    def _check_onnx_proto(self):
        print("===== Check if ONNX proto is valid. =====")
        return True

    def initialize(self, inputs, outputs):
        # Should we change the data type of inputs and outputs to acceptable ones, like Numpy?
        # Or we only accept ONNX compatible data types. Leave the data format conversion work out of ONNX Script.
        print("===== Graph and inputs, outputs are initialized. =====")

    def add_node(self, onnx_node):
        print("===== One node added. =====")

    def run_optimizers(self, skipped=None):
        if skipped:
            print("===== Skip some internal optimizers. =====")

        print("===== Run all internal optimizers. =====")

    def save_model_file(self, path=None):
        self._check_onnx_proto()
        print("===== Save the whole ONNX graph into a local file. =====")