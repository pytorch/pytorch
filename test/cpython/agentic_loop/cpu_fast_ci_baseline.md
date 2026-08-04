# CPU Fast CI Baseline

This baseline is for the agentic loop that runs CPython tests under Dynamo and
the Dynamo test suite in parallel, with CUDA tests skipped.

Command:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python agent_space/run_cpython_and_dynamo_timing.py --shards 32
```

Measured on 2026-05-20:

| Suite | Target | Wall time | Passed | Skipped | Expected failures |
|---|---|---:|---:|---:|---:|
| cpython_dynamo | `test/cpython/v3_13` with `PYTORCH_TEST_WITH_DYNAMO=1` | 72.9s | 2279 | 2461 | 1 |
| dynamo | `test/dynamo` | 217.7s | 9790 | 673 | 10 |

Combined wall time: 217.7s.

## Expected Failures

| Suite | Test | Failure |
|---|---|---|
| cpython_dynamo | `test/cpython/v3_13/test_builtin.py::BuiltinTest::test_callable` | `torch._dynamo.exc.Unsupported: Failed to trace builtin operator` |
| dynamo | `test/dynamo/test_dynamic_shapes.py::DynamicShapesReproTests::test_dynamo_set_recursion_limit_usage_dynamic_shapes` | `ModuleNotFoundError: No module named 'torchvision'` |
| dynamo | `test/dynamo/test_debug_utils.py::TestDynamoConfigOverrideIntegration::test_dynamo_config_override_warning` | `ModuleNotFoundError: No module named 'torchvision'` |
| dynamo | `test/dynamo/test_minifier.py::MinifierTestsCPU::test_after_dynamo_accuracy_error_cpu` | `AssertionError: 'AccuracyError' not found` |
| dynamo | `test/dynamo/test_minifier.py::MinifierTestsCPU::test_if_graph_minified_cpu` | `AssertionError: 'ReluCompileError' not found` |
| dynamo | `test/dynamo/test_compile.py::InPlaceCompilationTests::test_compile_frozen_module_error` | `ModuleNotFoundError: No module named 'torchvision'` |
| dynamo | `test/dynamo/test_repros.py::ReproTests::test_dynamo_set_recursion_limit_usage` | `ModuleNotFoundError: No module named 'torchvision'` |
| dynamo | `test/dynamo/test_structured_trace.py::StructuredTraceTest::test_recompiles` | `AssertionError: structured trace output mismatch` |
| dynamo | `test/dynamo/test_hooks.py::HooksTests::test_global_module_forward_pre_hook` | `ModuleNotFoundError: No module named 'torchvision'` |
| dynamo | `test/dynamo/test_utils.py::TestDynamoTimed::test_dynamo_timed` | `AssertionError: Dynamo timed output mismatch` |
| dynamo | `test/dynamo/test_compile.py::InPlaceCompilationTests::test_compile_frozen_module_inductor_error` | `ModuleNotFoundError: No module named 'torchvision'` |

Source run artifacts:

```text
agent_space/combined_cpython_dynamo_timing_20260520_124615/summary.json
```
