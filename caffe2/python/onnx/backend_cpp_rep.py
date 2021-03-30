## @package onnx
# Module caffe2.python.onnx.backend_rep_cpp






from onnx.backend.base import BackendRep, namedtupledict

# This is a wrapper around C++ Caffe2BackendRep,
# mainly to handle the different input and output types for convenience of Python
class Caffe2CppRep(BackendRep):
    def __init__(self, cpp_rep):
        super(Caffe2CppRep, self).__init__()
        self.__core = cpp_rep
        self.__external_outputs = cpp_rep.external_outputs()
        self.__external_inputs = cpp_rep.external_inputs()
        self.__uninitialized_inputs = cpp_rep.uninitialized_inputs()

    def init_net(self):
        return self.__core.init_net()

    def pred_net(self):
        return self.__core.pred_net()

    def external_outputs(self):
        return self.__core.external_outputs()

    def external_inputs(self):
        return self.__core.external_inputs()

    def run(self, inputs):
        output_values = None
        if isinstance(inputs, dict):
            output_values = self.__core.run(inputs)
        elif isinstance(inputs, list) or isinstance(inputs, tuple):
            if len(inputs) != len(self.__uninitialized_inputs):
                raise RuntimeError('Expected {} values for uninitialized '
                                   'graph inputs ({}), but got {}.'.format(
                                        len(self.__uninitialized_inputs),
                                        ', '.join(self.__uninitialized_inputs),
                                        len(inputs)))
            input_map = {}
            for k, v in zip(self.__uninitialized_inputs, inputs):
                input_map[k] = v
            output_values = self.__core.run(input_map)
        else:
            # single input
            output_values = self.__core.run([inputs])
        return namedtupledict('Outputs', self.__external_outputs)(*output_values)
