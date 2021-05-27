# -*- coding: utf-8 -*-

from torch.testing._internal.common_utils import run_tests

# Quantized Tensor
from quantization.core.test_quantized_tensor import TestQuantizedTensor  # noqa: F401
# Quantized Op
# TODO: merge test cases in quantization.test_quantized
from quantization.core.test_quantized_op import TestQuantizedOps  # noqa: F401
from quantization.core.test_quantized_op import TestQNNPackOps  # noqa: F401
from quantization.core.test_quantized_op import TestQuantizedLinear  # noqa: F401
from quantization.core.test_quantized_op import TestQuantizedConv  # noqa: F401
from quantization.core.test_quantized_op import TestDynamicQuantizedLinear  # noqa: F401
from quantization.core.test_quantized_op import TestComparatorOps  # noqa: F401
from quantization.core.test_quantized_op import TestPadding  # noqa: F401
from quantization.core.test_quantized_op import TestQuantizedEmbeddingOps  # noqa: F401
from quantization.core.test_quantized_op import TestDynamicQuantizedRNNOp  # noqa: F401
from quantization.core.test_quantized_functional import TestQuantizedFunctionalOps  # noqa: F401
from quantization.core.test_quantized_module import TestStaticQuantizedModule  # noqa: F401
from quantization.core.test_quantized_module import TestDynamicQuantizedModule  # noqa: F401


# Quantization specific fusion passes
from quantization.jit.test_fusion_passes import TestFusionPasses  # noqa: F401

# Module
from quantization.core.test_workflow_module import TestFakeQuantize  # noqa: F401
from quantization.core.test_workflow_module import TestObserver  # noqa: F401
# TODO: merge with TestObserver
# TODO: some tests belong to test_quantize.py, e.g. test_record_observer
from quantization.core.test_workflow_module import TestRecordHistogramObserver  # noqa: F401
from quantization.core.test_workflow_module import TestHistogramObserver  # noqa: F401
from quantization.core.test_workflow_module import TestDistributed  # noqa: F401

from quantization.core.test_workflow_ops import TestFakeQuantizeOps  # noqa: F401

# Workflow
# 1. Eager mode post training quantization
from quantization.eager.test_quantize_eager_ptq import TestPostTrainingStatic  # noqa: F401
from quantization.eager.test_quantize_eager_ptq import TestPostTrainingDynamic  # noqa: F401
from quantization.eager.test_quantize_eager_ptq import TestEagerModeActivationOps  # noqa: F401
from quantization.eager.test_quantize_eager_ptq import TestFunctionalModule  # noqa: F401
from quantization.eager.test_quantize_eager_ptq import TestQuantizeONNXExport  # noqa: F401

# Eager mode quantization aware training
from quantization.eager.test_quantize_eager_qat import TestQuantizationAwareTraining  # noqa: F401
from quantization.eager.test_quantize_eager_qat import TestQATActivationOps  # noqa: F401
from quantization.eager.test_quantize_eager_qat import TestConvBNQATModule  # noqa: F401

# Eager mode fusion passes
from quantization.eager.test_fusion import TestFusion  # noqa: F401

from quantization.eager.test_model_numerics import TestModelNumericsEager  # noqa: F401

from quantization.jit.test_deprecated_jit_quant import TestDeprecatedJitQuantized  # noqa: F401

# 2. Graph mode quantization
from quantization.jit.test_quantize_jit import TestQuantizeJit  # noqa: F401
from quantization.jit.test_quantize_jit import TestQuantizeJitPasses  # noqa: F401
from quantization.jit.test_quantize_jit import TestQuantizeJitOps  # noqa: F401
from quantization.jit.test_quantize_jit import TestQuantizeDynamicJitPasses  # noqa: F401
from quantization.jit.test_quantize_jit import TestQuantizeDynamicJitOps  # noqa: F401

# 3. GraphModule based graph mode quantization
try:
    from quantization.fx.test_quantize_fx import TestFuseFx  # noqa: F401
    from quantization.fx.test_quantize_fx import TestQuantizeFx  # noqa: F401
    from quantization.fx.test_quantize_fx import TestQuantizeFxOps  # noqa: F401
    from quantization.fx.test_quantize_fx import TestQuantizeFxModels  # noqa: F401
except ImportError:
    # In FBCode we separate FX out into a separate target for the sake of dev
    # velocity. These are covered by a separate test target `quantization_fx`
    pass

# Tooling: numeric_suite
from quantization.eager.test_numeric_suite_eager import TestEagerModeNumericSuite  # noqa: F401

try:
    from quantization.fx.test_numeric_suite_fx import TestFXGraphMatcher  # noqa: F401
    from quantization.fx.test_numeric_suite_fx import TestFXGraphMatcherModels  # noqa: F401
    from quantization.fx.test_numeric_suite_fx import TestFXNumericSuiteCoreAPIs  # noqa: F401
    from quantization.fx.test_numeric_suite_fx import TestFXNumericSuiteCoreAPIsModels  # noqa: F401
except ImportError:
    pass

# Backward Compatibility
from quantization.bc.test_backward_compatibility import TestSerialization  # noqa: F401

# Equalization
from quantization.eager.test_equalize_eager import TestEqualizeEager  # noqa: F401
# Bias Correction
from quantization.eager.test_bias_correction_eager import TestBiasCorrection  # noqa: F401

if __name__ == '__main__':
    run_tests()
