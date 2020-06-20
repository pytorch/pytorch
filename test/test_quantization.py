# -*- coding: utf-8 -*-

from torch.testing._internal.common_utils import run_tests

# Quantized Tensor
from quantization.test_quantized_tensor import TestQuantizedTensor  # noqa: F401
# Quantized Op
# TODO: merge test cases in quantization.test_quantized
from quantization.test_quantized_op import TestQuantizedOps  # noqa: F401
from quantization.test_quantized_op import TestQNNPackOps  # noqa: F401
from quantization.test_quantized_op import TestQuantizedLinear  # noqa: F401
from quantization.test_quantized_op import TestQuantizedConv  # noqa: F401
from quantization.test_quantized_op import TestDynamicQuantizedLinear  # noqa: F401
from quantization.test_quantized_op import TestComparatorOps  # noqa: F401
from quantization.test_quantized_op import TestPadding  # noqa: F401

# Quantized Functional
from quantization.test_quantized_functional import TestQuantizedFunctional  # noqa: F401

# Quantized Module
from quantization.test_quantized_module import TestStaticQuantizedModule  # noqa: F401
from quantization.test_quantized_module import TestDynamicQuantizedModule  # noqa: F401

# Quantization Aware Training
from quantization.test_qat_module import TestQATModule  # noqa: F401

# Quantization specific fusion passes
from quantization.test_fusion_passes import TestFusionPasses  # noqa: F401

# Module
# TODO: merge the fake quant per tensor and per channel test cases
# TODO: some of the tests are actually operator tests, e.g. test_forward_per_tensor, and
# should be moved to test_quantized_op
from quantization.test_workflow_module import TestFakeQuantizePerTensor  # noqa: F401
from quantization.test_workflow_module import TestFakeQuantizePerChannel  # noqa: F401
from quantization.test_workflow_module import TestObserver  # noqa: F401
# TODO: merge with TestObserver
# TODO: some tests belong to test_quantize.py, e.g. test_record_observer
from quantization.test_workflow_module import TestRecordHistogramObserver  # noqa: F401
from quantization.test_workflow_module import TestDistributed  # noqa: F401

# Workflow
# 1. Eager mode quantization
from quantization.test_quantize import TestPostTrainingStatic  # noqa: F401
from quantization.test_quantize import TestPostTrainingDynamic  # noqa: F401
from quantization.test_quantize import TestQuantizationAwareTraining  # noqa: F401

# TODO: merge with other tests in test_quantize.py?
from quantization.test_quantize import TestFunctionalModule  # noqa: F401
from quantization.test_quantize import TestFusion  # noqa: F401
from quantization.test_quantize import TestModelNumerics  # noqa: F401
# 2. Graph mode quantization
from quantization.test_quantize_script import TestQuantizeScriptJitPasses  # noqa: F401
from quantization.test_quantize_script import TestQuantizeScriptPTSQOps  # noqa: F401
from quantization.test_quantize_script import TestQuantizeDynamicScriptJitPasses  # noqa: F401
from quantization.test_quantize_script import TestQuantizeScriptPTDQOps  # noqa: F401
from quantization.test_quantize_script import TestQuantizeScript  # noqa: F401
from quantization.test_quantize_script import TestQuantizeScriptJit  # noqa: F401

# Tooling: numric_suite
from quantization.test_numeric_suite import TestEagerModeNumericSuite  # noqa: F401

# Backward Compatibility
from quantization.test_backward_compatibility import TestSerialization  # noqa: F401

if __name__ == '__main__':
    run_tests()
