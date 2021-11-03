# Owner(s): ["module: autograd"]

from torch.testing._internal.common_utils import TestCase, run_tests

import torch

import pkgutil
import importlib

from types import ModuleType

try:
    import tensorrt as trt
except Exception:
    trt = {}

class TestPublicBindings(TestCase):
    def test_valid_module_attribute(self):
        # No new entry should EVER be added to this set, only removed!
        missing_all_attr_allowlist = set([
            'torch.ao', 'torch.ao.nn', 'torch.ao.nn.sparse', 'torch.ao.ns',
            'torch.ao.ns.fx', 'torch.ao.quantization', 'torch.ao.quantization.fx',
            'torch.ao.quantization.fx.backend_config_dict', 'torch.ao.sparsity',
            'torch.ao.sparsity.experimental', 'torch.ao.sparsity.experimental.pruner',
            'torch.ao.sparsity.scheduler', 'torch.ao.sparsity.sparsifier', 'torch.backends',
            'torch.backends.cuda', 'torch.backends.cudnn', 'torch.backends.mkl', 'torch.backends.mkldnn',
            'torch.backends.openmp', 'torch.backends.quantized', 'torch.backends.xnnpack', 'torch.contrib',
            'torch.cpu', 'torch.cpu.amp', 'torch.cuda', 'torch.cuda.amp', 'torch.distributed',
            'torch.distributed.algorithms', 'torch.distributed.algorithms.model_averaging',
            'torch.distributed.autograd', 'torch.distributed.elastic', 'torch.distributed.elastic.agent',
            'torch.distributed.elastic.events', 'torch.distributed.elastic.metrics',
            'torch.distributed.elastic.multiprocessing', 'torch.distributed.elastic.multiprocessing.errors',
            'torch.distributed.elastic.timer', 'torch.distributed.elastic.utils',
            'torch.distributed.elastic.utils.data', 'torch.distributed.pipeline','torch.distributed.rpc',
            'torch.for_onnx', 'torch.futures', 'torch.fx', 'torch.fx.experimental', 'torch.fx.experimental.fx_acc',
            'torch.fx.experimental.unification', 'torch.fx.experimental.unification.multipledispatch',
            'torch.fx.passes', 'torch.jit', 'torch.jit.mobile', 'torch.linalg', 'torch.nn', 'torch.nn.backends',
            'torch.nn.intrinsic', 'torch.nn.intrinsic.qat', 'torch.nn.intrinsic.quantized',
            'torch.nn.intrinsic.quantized.dynamic', 'torch.nn.qat', 'torch.nn.quantizable', 'torch.nn.quantized',
            'torch.nn.quantized.dynamic', 'torch.nn.utils', 'torch.onnx', 'torch.optim', 'torch.package',
            'torch.package.analyze', 'torch.profiler', 'torch.quantization', 'torch.quantization.fx', 'torch.testing',
            'torch.utils', 'torch.utils.backcompat', 'torch.utils.benchmark', 'torch.utils.benchmark.examples',
            'torch.utils.benchmark.op_fuzzers', 'torch.utils.benchmark.utils',
            'torch.utils.benchmark.utils.valgrind_wrapper', 'torch.utils.bottleneck', 'torch.utils.data.communication',
            'torch.utils.data.datapipes', 'torch.utils.data.datapipes.utils', 'torch.utils.hipify',
            'torch.utils.model_dump', 'torch.utils.tensorboard',
        ])

        # No new entry should EVER be added to this set, only removed!
        missing_module_attr_allowlist = set([
            'torch.SUM', 'torch.ScriptDictIterator', 'torch.ScriptClass', 'torch.fft.fftn', 'torch.Future',
            'torch.IntType', 'torch.ScriptObject', 'torch.Value', 'torch.CallStack', 'torch.ComplexType',
            'torch.ErrorReport', 'torch.clear_autocast_cache', 'torch.Use', 'torch.utils.data.argument_validation',
            'torch.CompleteArgumentSpec', 'torch.fft.rfft2', 'torch.AliasDb', 'torch.LiteScriptModule',
            'torch.fft.ifftn', 'torch.randn', 'torch.special.sinc', 'torch.FunctionSchema',
            'torch.MobileOptimizerType', 'torch.REMOVE_DROPOUT', 'torch.FutureType', 'torch.fft.rfftn',
            'torch.special.i1e', 'torch.Capsule', 'torch.special.softmax', 'torch.special.gammaln',
            'torch.fft.ifftshift', 'torch.INSERT_FOLD_PREPACK_OPS', 'torch.special.digamma', 'torch.EnumType',
            'torch.ClassType', 'torch.NoneType', 'torch.fft.fftshift', 'torch.fft.ifft2', 'torch.CONV_BN_FUSION',
            'torch.BenchmarkExecutionStats', 'torch.BufferDict', 'torch.TupleType', 'torch.special.log1p',
            'torch.NoopLogger', 'torch.FUSE_ADD_RELU', 'torch.OptionalType', 'torch.ModuleDict', 'torch.fft.fft2',
            'torch.special.log_softmax', 'torch.TracingState', 'torch.special.exp2', 'torch.GraphExecutorState',
            'torch.ConcreteModuleType', 'torch.matmul', 'torch.InterfaceType', 'torch.special.ndtr',
            'torch.special.round', 'torch.DeviceObjType', 'torch.BenchmarkConfig', 'torch.special.psi',
            'torch.ConcreteModuleTypeBuilder', 'torch.AggregationType', 'torch.rand',
            'torch.SerializationStorageContext', 'torch.fft.fftfreq', 'torch.PyTorchFileReader', 'torch.Block',
            'torch.ScriptObjectProperty', 'torch.ScriptModuleSerializer', 'torch.special.xlog1py',
            'torch.ExecutionPlan', 'torch.ParameterDict', 'torch.special.logit', 'torch.ListType',
            'torch.StaticModule', 'torch.AnyType', 'torch.fft.rfftfreq', 'torch.Node', 'torch.special.erf',
            'torch.Type', 'torch.IODescriptor', 'torch.fft.rfft', 'torch.Graph', 'torch.special.zeta',
            'torch.OperatorInfo', 'torch.ScriptMethod', 'torch.Argument', 'torch.ScriptDictKeyIterator',
            'torch.ScriptListIterator', 'torch.ThroughputBenchmark', 'torch.ArgumentSpec', 'torch.special.expit',
            'torch.special.gammainc', 'torch.special.polygamma', 'torch.special.erfinv', 'torch.special.entr',
            'torch.TensorType', 'torch.special.ndtri', 'torch.AVG', 'torch.ScriptDict', 'torch.fft.irfftn',
            'torch.autocast_increment_nesting', 'torch.special.expm1', 'torch.DictType', 'torch.NumberType',
            'torch.HOIST_CONV_PACKED_PARAMS', 'torch.Code', 'torch.fft.hfft', 'torch.UnionType', 'torch.special.i0',
            'torch.StringType', 'torch.fft.ifft', 'torch.special.i0e', 'torch.special.xlogy', 'torch.chunk',
            'torch.fft.ihfft', 'torch.Gradient', 'torch.FloatType', 'torch.RRefType', 'torch.ScriptList',
            'torch.FileCheck', 'torch.special.erfcx', 'torch.special.gammaincc', 'torch.InferredType',
            'torch.ScriptClassFunction', 'torch.fft.irfft', 'torch.PyTorchFileWriter', 'torch.DeepCopyMemoTable',
            'torch.autocast_decrement_nesting', 'torch.ScriptModule', 'torch.set_printoptions', 'torch.fft.irfft2',
            'torch.PyObjectType', 'torch.special.erfc', 'torch.StreamObjType', 'torch.special.logsumexp',
            'torch.lobpcg', 'torch.DeserializationStorageContext', 'torch.stack', 'torch.LockingLogger',
            'torch.special.i1', 'torch.Generator', 'torch.BoolType', 'torch.DisableTorchFunction', 'torch.fft.fft',
            'torch.special.multigammaln',
        ])

        # No new entry should EVER be added to this set, only removed!
        missing_obj_attr_allowlist = set(['torch.utils.data.datapipes.iter.DFIterDataPipe'])

        def is_not_internal(modname):
            split_name = modname.split(".")
            for name in split_name:
                if name[0] == "_":
                    return False
            return True

        # Allow this script to run when
        #  - Built with USE_DISTRIBUTED=0
        #  - TensorRT is not installed
        #  - Until torch.utils.ffi module is removed
        def cannot_be_skipped(modname):
            if "distributed" in modname and not torch.distributed.is_available():
                return False
            if "fx2trt" in modname and not hasattr(trt, "__version__"):
                return False
            if modname == "torch.utils.ffi":
                return False
            return True

        missing_all_attr = set()
        missing_module_attr = set()
        missing_obj_attr = set()

        def error_handler(bad_pkg):
            if is_not_internal(bad_pkg) and cannot_be_skipped(bad_pkg):
                raise RuntimeError(f"Failed to import public package {bad_pkg}")

        def process(modname, ispkg):
            if ispkg and is_not_internal(modname):
                try:
                    mod = importlib.import_module(modname)
                except Exception as e:
                    if cannot_be_skipped(modname):
                        raise e
                    return

                if not hasattr(mod, "__all__"):
                    missing_all_attr.add(modname)
                    return
                for el_name in mod.__all__:
                    if not hasattr(mod, el_name):
                        missing_obj_attr.add(f"{modname}.{el_name}")
                        return
                    obj = getattr(mod, el_name)
                    if isinstance(obj, ModuleType):
                        return
                    if getattr(obj, "__module__", None) is None:
                        missing_module_attr.add(f"{modname}.{el_name}")
                    elif obj.__module__ != modname:
                        m = obj.__module__
                        if m.startswith("torch") and not is_not_internal(m):
                            missing_module_attr.add(f"{modname}.{el_name}")

        process("torch", True)
        for _, modname, ispkg in pkgutil.walk_packages(path=torch.__path__,
                                                       prefix=torch.__name__ + '.',
                                                       onerror=error_handler):
            process(modname, ispkg)

        output = []

        # Generate error for missing `__all__` attribute on a module
        unexpected_missing = missing_all_attr - missing_all_attr_allowlist
        if unexpected_missing:
            mods = ", ".join(unexpected_missing)
            output.append(f"\nYou added the following module(s) to the PyTorch namespace '{mods}' "
                          "but they have no `__all__` attribute in a doc .rst file. You should use "
                          "this attribute to specify which functions are public.")
        unexpected_not_missing = missing_all_attr_allowlist - missing_all_attr
        if unexpected_not_missing:
            mods = ", ".join(unexpected_not_missing)
            output.append(f"\nThank you for adding the missing `__all__` for '{mods}', please update "
                          "the 'missing_all_attr_allowlist' in 'torch/test/test_public_bindings.py' by removing "
                          "the module(s) you fixed to make sure we do not regress on this in the future.")

        # Generate error for missing/wrong `__module__` attribute on a public API
        unexpected_missing = missing_module_attr - missing_module_attr_allowlist
        if unexpected_missing:
            mods = ", ".join(unexpected_missing)
            output.append(f"\nYou added the following function/class(es) to PyTorch '{mods}' "
                          "but they have no or incorrect `__module__` attribute. This attribute "
                          "must point to the module that exposes these functions as public API (as "
                          "defined using the `__all__` attribute on the module).")
        unexpected_not_missing = missing_module_attr_allowlist - missing_module_attr
        if unexpected_not_missing:
            mods = ", ".join(unexpected_not_missing)
            output.append(f"\nThank you for fixing the `__module__` attribute for '{mods}', please update "
                          "the 'missing_module_attr_allowlist' in 'torch/test/test_public_bindings.py' by "
                          "removing the function/class(es) you fixed to make sure we do not regress on this "
                          "in the future.")

        # Generate error for missing function/class that is listed in `__all__`
        unexpected_missing = missing_obj_attr - missing_obj_attr_allowlist
        if unexpected_missing:
            mods = ", ".join(unexpected_missing)
            output.append(f"\nYou added the following function/class(es) to PyTorch '{mods}' "
                          "but, while they are part of the list of public APIs for their module "
                          "(as described by `__all__`), this object does not exist on this module.")
        unexpected_not_missing = missing_obj_attr_allowlist - missing_obj_attr
        if unexpected_not_missing:
            mods = ", ".join(unexpected_not_missing)
            output.append(f"\nThank you for fixing the existence of '{mods}', please update the "
                          "'missing_obj_attr_allowlist' in 'torch/test/test_public_bindings.py' by "
                          "removing the function/class(es) you fixed to make sure we do not regress on this "
                          "in the future.")

        self.assertFalse(output, msg="Some Error where detected in the namespace attributes: " + "\n".join(output))

    def test_no_new_bindings(self):
        """
        This test aims to stop the introduction of new JIT bindings into torch._C
        whose names do not start with _. Such bindings are made available as
        torch.XXX, which may not be desirable.

        If your change causes this test to fail, add your new binding to a relevant
        submodule of torch._C, such as torch._C._jit (or other relevant submodule of
        torch._C). If your binding really needs to be available as torch.XXX, add it
        to torch._C and add it to the allowlist below.

        If you have removed a binding, remove it from the allowlist as well.
        """
        # This allowlist contains every binding in torch._C that is copied into torch at
        # the time of writing. It was generated with
        #
        #   {elem for elem in dir(torch._C) if not elem.startswith("_")}
        #
        torch_C_allowlist_superset = {
            "AggregationType",
            "AliasDb",
            "AnyType",
            "Argument",
            "ArgumentSpec",
            "autocast_decrement_nesting",
            "autocast_increment_nesting",
            "AVG",
            "BenchmarkConfig",
            "BenchmarkExecutionStats",
            "BFloat16StorageBase",
            "Block",
            "BoolStorageBase",
            "BoolType",
            "BufferDict",
            "ByteStorageBase",
            "CallStack",
            "Capsule",
            "CharStorageBase",
            "ClassType",
            "clear_autocast_cache",
            "Code",
            "CompilationUnit",
            "CompleteArgumentSpec",
            "ComplexDoubleStorageBase",
            "ComplexFloatStorageBase",
            "ComplexType",
            "ConcreteModuleType",
            "ConcreteModuleTypeBuilder",
            "CONV_BN_FUSION",
            "cpp",
            "CudaBFloat16StorageBase",
            "CudaBFloat16TensorBase",
            "CudaBFloat16TensorBase",
            "CudaBoolStorageBase",
            "CudaBoolTensorBase",
            "CudaBoolTensorBase",
            "CudaByteStorageBase",
            "CudaByteTensorBase",
            "CudaByteTensorBase",
            "CudaCharStorageBase",
            "CudaCharTensorBase",
            "CudaCharTensorBase",
            "CudaComplexDoubleStorageBase",
            "CudaComplexDoubleTensorBase",
            "CudaComplexDoubleTensorBase",
            "CudaComplexFloatStorageBase",
            "CudaComplexFloatTensorBase",
            "CudaComplexFloatTensorBase",
            "CudaDoubleStorageBase",
            "CudaDoubleTensorBase",
            "CudaDoubleTensorBase",
            "CudaFloatStorageBase",
            "CudaFloatTensorBase",
            "CudaHalfStorageBase",
            "CudaHalfTensorBase",
            "CudaIntStorageBase",
            "CudaIntTensorBase",
            "CudaIntTensorBase",
            "CudaLongStorageBase",
            "CudaLongTensorBase",
            "CudaLongTensorBase",
            "CudaShortStorageBase",
            "CudaShortTensorBase",
            "CudaShortTensorBase",
            "DeepCopyMemoTable",
            "default_generator",
            "DeserializationStorageContext",
            "device",
            "DeviceObjType",
            "DictType",
            "DisableTorchFunction",
            "DoubleStorageBase",
            "dtype",
            "EnumType",
            "ErrorReport",
            "ExecutionPlan",
            "FatalError",
            "FileCheck",
            "finfo",
            "FloatStorageBase",
            "FloatType",
            "fork",
            "FunctionSchema",
            "FUSE_ADD_RELU",
            "Future",
            "FutureType",
            "Generator",
            "get_autocast_cpu_dtype",
            "get_default_dtype",
            "get_num_interop_threads",
            "get_num_threads",
            "Gradient",
            "Graph",
            "GraphExecutorState",
            "HalfStorageBase",
            "has_cuda",
            "has_cudnn",
            "has_lapack",
            "has_mkl",
            "has_mkldnn",
            "has_mlc",
            "has_openmp",
            "has_spectral",
            "HOIST_CONV_PACKED_PARAMS",
            "iinfo",
            "import_ir_module_from_buffer",
            "import_ir_module",
            "InferredType",
            "init_num_threads",
            "INSERT_FOLD_PREPACK_OPS",
            "InterfaceType",
            "IntStorageBase",
            "IntType",
            "IODescriptor",
            "is_anomaly_enabled",
            "is_autocast_cache_enabled",
            "is_autocast_cpu_enabled",
            "is_autocast_enabled",
            "is_grad_enabled",
            "is_inference_mode_enabled",
            "JITException",
            "layout",
            "ListType",
            "LiteScriptModule",
            "LockingLogger",
            "LoggerBase",
            "LongStorageBase",
            "memory_format",
            "merge_type_from_type_comment",
            "MobileOptimizerType",
            "ModuleDict",
            "Node",
            "NoneType",
            "NoopLogger",
            "NumberType",
            "OperatorInfo",
            "OptionalType",
            "ParameterDict",
            "parse_ir",
            "parse_schema",
            "parse_type_comment",
            "PyObjectType",
            "PyTorchFileReader",
            "PyTorchFileWriter",
            "QInt32StorageBase",
            "QInt8StorageBase",
            "qscheme",
            "QUInt4x2StorageBase",
            "QUInt2x4StorageBase",
            "QUInt8StorageBase",
            "read_vitals",
            "REMOVE_DROPOUT",
            "RRefType",
            "ScriptClass",
            "ScriptClassFunction",
            "ScriptDict",
            "ScriptDictIterator",
            "ScriptDictKeyIterator",
            "ScriptList",
            "ScriptListIterator",
            "ScriptFunction",
            "ScriptMethod",
            "ScriptModule",
            "ScriptModuleSerializer",
            "ScriptObject",
            "ScriptObjectProperty",
            "SerializationStorageContext",
            "set_anomaly_enabled",
            "set_autocast_cache_enabled",
            "set_autocast_cpu_dtype",
            "set_autocast_cpu_enabled",
            "set_autocast_enabled",
            "set_flush_denormal",
            "set_num_interop_threads",
            "set_num_threads",
            "set_vital",
            "ShortStorageBase",
            "Size",
            "StaticModule",
            "Stream",
            "StreamObjType",
            "StringType",
            "SUM",
            "TensorType",
            "ThroughputBenchmark",
            "TracingState",
            "TupleType",
            "Type",
            "unify_type_list",
            "UnionType",
            "Use",
            "Value",
            "autocast_decrement_nesting",
            "autocast_increment_nesting",
            "clear_autocast_cache",
            "cpp",
            "default_generator",
            "device",
            "dtype",
            "finfo",
            "fork",
            "get_default_dtype",
            "get_num_interop_threads",
            "get_num_threads",
            "has_cuda",
            "has_cudnn",
            "has_lapack",
            "has_mkl",
            "has_mkldnn",
            "has_mlc",
            "has_openmp",
            "iinfo",
            "import_ir_module",
            "import_ir_module_from_buffer",
            "init_num_threads",
            "is_anomaly_enabled",
            "is_autocast_enabled",
            "is_grad_enabled",
            "layout",
            "memory_format",
            "merge_type_from_type_comment",
            "parse_ir",
            "parse_schema",
            "parse_type_comment",
            "qscheme",
            "set_anomaly_enabled",
            "set_autocast_enabled",
            'set_autocast_gpu_dtype',
            'get_autocast_gpu_dtype',
            "set_flush_denormal",
            "set_num_interop_threads",
            "set_num_threads",
            "unify_type_list",
            "vitals_enabled",

            "wait",
        }
        torch_C_bindings = {elem for elem in dir(torch._C) if not elem.startswith("_")}

        # Check that the torch._C bindings are all in the allowlist. Since
        # bindings can change based on how PyTorch was compiled (e.g. with/without
        # CUDA), the two may not be an exact match but the bindings should be
        # a subset of the allowlist.
        difference = torch_C_bindings.difference(torch_C_allowlist_superset)
        msg = f"torch._C had bindings that are not present in the allowlist:\n{difference}"
        self.assertTrue(torch_C_bindings.issubset(torch_C_allowlist_superset), msg)


if __name__ == '__main__':
    run_tests()
