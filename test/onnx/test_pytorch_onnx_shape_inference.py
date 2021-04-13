import unittest
import torch

import copy

import test_pytorch_onnx_onnxruntime
from test_pytorch_onnx_onnxruntime import TestONNXRuntime
from torch.onnx import utils, OperatorExportTypes, TrainingMode
from torch.onnx.utils import _validate_dynamic_axes
from torch.onnx.symbolic_helper import (_set_opset_version, _set_operator_export_type,
                                        _set_onnx_shape_inference, _set_training_mode,
                                        _is_tensor_list, _is_tensor, _is_none)


def verify_inferred_shape(graph):
    # Check every node in graph has type properly assigned.
    for n in graph.nodes():
        for out in n.outputs():
            if not _is_tensor_list(out) and not _is_tensor(out) and not _is_none(out):
                raise RuntimeError("Output of node is neither type Tensor nor type list of Tensor: ", out)
            if _is_tensor(out) and out.type().scalarType() is None:
                raise RuntimeError("Output of node does not have type assigned", out)
            if _is_tensor(out) and out.type().dim() is None:
                raise RuntimeError("Output of node does not have shape assigned", out)


def run_model_test(self, model, batch_size=2, state_dict=None,
                   input=None, use_gpu=True, rtol=0.001, atol=1e-7,
                   example_outputs=None, do_constant_folding=True,
                   dynamic_axes=None, test_with_inputs=None,
                   input_names=None, output_names=None,
                   fixed_batch_size=False):
    model.eval()

    if input is None:
        input = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    with torch.no_grad():
        if isinstance(input, torch.Tensor):
            input = (input,)
        # In-place operators will update input tensor data as well.
        # Thus inputs are replicated before every forward call.
        input_copy = copy.deepcopy(input)
        output = model(*input_copy)
        if isinstance(output, torch.Tensor):
            output = (output,)

        _set_opset_version(self.opset_version)
        _set_operator_export_type(OperatorExportTypes.ONNX)
        _set_onnx_shape_inference(True)
        _set_training_mode(False)
        if dynamic_axes is None:
            dynamic_axes = {}
        _validate_dynamic_axes(dynamic_axes, model, input_names, output_names)

        input_copy = copy.deepcopy(input)
        graph, _, _ = utils._model_to_graph(model, input_copy,
                                            input_names=input_names,
                                            output_names=output_names,
                                            operator_export_type=OperatorExportTypes.ONNX,
                                            example_outputs=output,
                                            do_constant_folding=do_constant_folding,
                                            training=TrainingMode.EVAL,
                                            dynamic_axes=dynamic_axes)
        verify_inferred_shape(graph)


if __name__ == '__main__':
    TestONNXRuntime.opset_version = 12
    test_pytorch_onnx_onnxruntime.run_model_test = run_model_test

    unittest.main()
