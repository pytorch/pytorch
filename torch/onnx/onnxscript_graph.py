
class OSGraph(object):
    def __init__(self):
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