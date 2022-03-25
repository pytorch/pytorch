# -*- coding: utf-8 -*-
# Owner(s): ["module: autograd"]

from torch.testing._internal.common_utils import TestCase, run_tests
import pkgutil
import torch
import sys
from typing import Callable
import inspect

class TestPublicBindings(TestCase):
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

    def test_correct_module_names(self):
        '''
        An API is considered public, if  its  `__module__` starts with `torch.`
        and there is no name in `__module__` or the object itself that starts with “_”.
        Each public package should either:
        - (preferred) Define `__all__` and all callables and classes in there must have their
         `__module__` start with the current submodule's path. Things not in `__all__` should
          NOT have their `__module__` start with the current submodule.
        - (for simple python-only modules) Not define `__all__` and all the elements in `dir(submod)` must have their
          `__module__` that start with the current submodule.
        '''
        failure_list = []

        def test_module(modname):
            is_a_private_module = False
            split_strs = modname.split('.')
            mod = sys.modules.get(modname)
            for elem in split_strs:
                if elem.startswith("_"):
                    is_a_private_module = True
                    break

            if is_a_private_module:
                return

            # verifies that each API has the correct module name and naming semantics
            # depending on whether it's public or private
            def looks_public(elem, modname, mod, private_api):
                obj = getattr(mod, elem)
                if not (isinstance(obj, Callable) or inspect.isclass(obj)):
                    return
                elem_module = getattr(obj, '__module__', None)
                elem_modname_starts_with_modname = elem_module is not None and elem_module.startswith(modname)
                if private_api:
                    # elem's name must begin with an `_` and it's module name
                    # should NOT start with it's current module since it's a private API
                    if elem_modname_starts_with_modname:
                        failure_list.append((modname, elem, elem_module))
                else:
                    # elem's name must NOT begin with an `_` and it's module name
                    # SHOULD start with it's current module since it's a public API
                    if elem.startswith('_') and not elem_modname_starts_with_modname:
                        failure_list.append((modname, elem, elem_module))

            if hasattr(modname, '__all__'):
                public_api = mod.__all__
                all_api = dir(modname)
                for elem in all_api:
                    looks_public(elem, modname, elem not in public_api)
            else:
                all_api = dir(mod)
                for elem in all_api:
                    looks_public(elem, modname, mod, elem.startswith('_'))

        for _, modname, ispkg in pkgutil.walk_packages(path=torch.__path__, prefix=torch.__name__ + '.'):
            test_module(modname)

        test_module('torch')
        self.assertExpected("\n".join(map(str, failure_list)), 'api_checks')

if __name__ == '__main__':
    run_tests()
