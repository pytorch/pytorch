# Owner(s): ["oncall: quantization"]

import logging
from torch.testing._internal.common_utils import run_tests

# Quantization core tests. These include tests for
# - quantized kernels
# - quantized functional operators
# - quantized workflow modules
# - quantized workflow operators
# - quantized tensor

# 1. Quantized Kernels
# TODO: merge the different quantized op tests into one test class
from quantization.core.test_quantized_op import TestQuantizedOps  # noqa: F401
from quantization.core.test_quantized_op import TestQNNPackOps  # noqa: F401
from quantization.core.test_quantized_op import TestQuantizedLinear  # noqa: F401
from quantization.core.test_quantized_op import TestQuantizedConv  # noqa: F401
from quantization.core.test_quantized_op import TestDynamicQuantizedOps  # noqa: F401
from quantization.core.test_quantized_op import TestComparatorOps  # noqa: F401
from quantization.core.test_quantized_op import TestPadding  # noqa: F401
from quantization.core.test_quantized_op import TestQuantizedEmbeddingOps  # noqa: F401
# 2. Quantized Functional/Workflow Ops
from quantization.core.test_quantized_functional import TestQuantizedFunctionalOps  # noqa: F401
from quantization.core.test_workflow_ops import TestFakeQuantizeOps  # noqa: F401
from quantization.core.test_workflow_ops import TestFusedObsFakeQuant  # noqa: F401
# 3. Quantized Tensor
from quantization.core.test_quantized_tensor import TestQuantizedTensor  # noqa: F401
# 4. Modules
from quantization.core.test_workflow_module import TestFakeQuantize  # noqa: F401
from quantization.core.test_workflow_module import TestObserver  # noqa: F401
from quantization.core.test_quantized_module import TestStaticQuantizedModule  # noqa: F401
from quantization.core.test_quantized_module import TestDynamicQuantizedModule  # noqa: F401
from quantization.core.test_quantized_module import TestReferenceQuantizedModule  # noqa: F401
from quantization.core.test_workflow_module import TestRecordHistogramObserver  # noqa: F401
from quantization.core.test_workflow_module import TestHistogramObserver  # noqa: F401
from quantization.core.test_workflow_module import TestDistributed  # noqa: F401
from quantization.core.test_workflow_module import TestFusedObsFakeQuantModule  # noqa: F401
from quantization.core.test_backend_config import TestBackendConfig  # noqa: F401
from quantization.core.test_utils import TestUtils  # noqa: F401

# Eager Mode Workflow. Tests for the functionality of APIs and different features implemented
# using eager mode.

# 1. Eager mode post training quantization
from quantization.eager.test_quantize_eager_ptq import TestQuantizeEagerPTQStatic  # noqa: F401
from quantization.eager.test_quantize_eager_ptq import TestQuantizeEagerPTQDynamic  # noqa: F401
from quantization.eager.test_quantize_eager_ptq import TestQuantizeEagerOps  # noqa: F401
# 2. Eager mode quantization aware training
from quantization.eager.test_quantize_eager_qat import TestQuantizeEagerQAT  # noqa: F401
from quantization.eager.test_quantize_eager_qat import TestQuantizeEagerQATNumerics  # noqa: F401
# 3. Eager mode fusion passes
from quantization.eager.test_fuse_eager import TestFuseEager  # noqa: F401
# 4. Testing model numerics between quantized and FP32 models
from quantization.eager.test_model_numerics import TestModelNumericsEager  # noqa: F401
# 5. Tooling: numeric_suite
from quantization.eager.test_numeric_suite_eager import TestNumericSuiteEager  # noqa: F401
# 6. Equalization and Bias Correction
from quantization.eager.test_equalize_eager import TestEqualizeEager  # noqa: F401
from quantization.eager.test_bias_correction_eager import TestBiasCorrectionEager  # noqa: F401


log = logging.getLogger(__name__)
# FX GraphModule Graph Mode Quantization. Tests for the functionality of APIs and different features implemented
# using fx quantization.
try:
    from quantization.fx.test_quantize_fx import TestFuseFx  # noqa: F401
    from quantization.fx.test_quantize_fx import TestQuantizeFx  # noqa: F401
    from quantization.fx.test_quantize_fx import TestQuantizeFxOps  # noqa: F401
    from quantization.fx.test_quantize_fx import TestQuantizeFxModels  # noqa: F401
    from quantization.fx.test_subgraph_rewriter import TestSubgraphRewriter  # noqa: F401
except ImportError as e:
    # In FBCode we separate FX out into a separate target for the sake of dev
    # velocity. These are covered by a separate test target `quantization_fx`
    log.warning(e)  # noqa:G200

try:
    from quantization.fx.test_numeric_suite_fx import TestFXGraphMatcher  # noqa: F401
    from quantization.fx.test_numeric_suite_fx import TestFXGraphMatcherModels  # noqa: F401
    from quantization.fx.test_numeric_suite_fx import TestFXNumericSuiteCoreAPIs  # noqa: F401
    from quantization.fx.test_numeric_suite_fx import TestFXNumericSuiteNShadows  # noqa: F401
    from quantization.fx.test_numeric_suite_fx import TestFXNumericSuiteCoreAPIsModels  # noqa: F401
except ImportError as e:
    log.warning(e)  # noqa:G200

# Test the model report module
try:
    from quantization.fx.test_model_report_fx import TestFxModelReportDetector  # noqa: F401
    from quantization.fx.test_model_report_fx import TestFxModelReportObserver      # noqa: F401
    from quantization.fx.test_model_report_fx import TestFxModelReportDetectDynamicStatic  # noqa: F401
    from quantization.fx.test_model_report_fx import TestFxModelReportClass  # noqa: F401
    from quantization.fx.test_model_report_fx import TestFxDetectInputWeightEqualization  # noqa: F401
    from quantization.fx.test_model_report_fx import TestFxDetectOutliers  # noqa: F401
    from quantization.fx.test_model_report_fx import TestFxModelReportVisualizer  # noqa: F401
except ImportError as e:
    log.warning(e)  # noqa:G200

# Equalization for FX mode
try:
    from quantization.fx.test_equalize_fx import TestEqualizeFx  # noqa: F401
except ImportError as e:
    log.warning(e)  # noqa:G200

# Backward Compatibility. Tests serialization and BC for quantized modules.
try:
    from quantization.bc.test_backward_compatibility import TestSerialization  # noqa: F401
except ImportError as e:
    log.warning(e)  # noqa:G200

# JIT Graph Mode Quantization
from quantization.jit.test_quantize_jit import TestQuantizeJit  # noqa: F401
from quantization.jit.test_quantize_jit import TestQuantizeJitPasses  # noqa: F401
from quantization.jit.test_quantize_jit import TestQuantizeJitOps  # noqa: F401
from quantization.jit.test_quantize_jit import TestQuantizeDynamicJitPasses  # noqa: F401
from quantization.jit.test_quantize_jit import TestQuantizeDynamicJitOps  # noqa: F401
# Quantization specific fusion passes
from quantization.jit.test_fusion_passes import TestFusionPasses  # noqa: F401
from quantization.jit.test_deprecated_jit_quant import TestDeprecatedJitQuantized  # noqa: F401

# AO Migration tests
from quantization.ao_migration.test_quantization import TestAOMigrationQuantization  # noqa: F401
from quantization.ao_migration.test_ao_migration import TestAOMigrationNNQuantized  # noqa: F401
from quantization.ao_migration.test_ao_migration import TestAOMigrationNNIntrinsic  # noqa: F401
try:
    from quantization.ao_migration.test_quantization_fx import TestAOMigrationQuantizationFx  # noqa: F401
except ImportError as e:
    log.warning(e)  # noqa:G200

# Experimental functionality
try:
    from quantization.core.experimental.test_bits import TestBitsCPU  # noqa: F401
except ImportError as e:
    log.warning(e)  # noqa:G200
try:
    from quantization.core.experimental.test_bits import TestBitsCUDA  # noqa: F401
except ImportError as e:
    log.warning(e)  # noqa:G200
try:
    from quantization.core.experimental.test_floatx import TestFloat8DtypeCPU  # noqa: F401
except ImportError as e:
    log.warning(e)  # noqa:G200
try:
    from quantization.core.experimental.test_floatx import TestFloat8DtypeCUDA  # noqa: F401
except ImportError as e:
    log.warning(e)  # noqa:G200
try:
    from quantization.core.experimental.test_floatx import TestFloat8DtypeCPUOnlyCPU  # noqa: F401
except ImportError as e:
    log.warning(e)  # noqa:G200

if __name__ == '__main__':
    run_tests()
