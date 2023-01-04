# -*- coding: utf-8 -*-
# Owner(s): ["module: autograd"]

from torch.testing._internal.common_utils import TestCase, run_tests, IS_WINDOWS
import pkgutil
import torch
import sys
from typing import Callable
import inspect
import json
import os
import unittest


# TODO(jansel): we should remove this workaround once this is fixed:
# https://github.com/pytorch/pytorch/issues/86619
NOT_IMPORTED_WHEN_TEST_WRITTEN = {
    "torch.fx.experimental.normalize",
    "torch.fx.experimental.proxy_tensor",
    "torch.fx.experimental.schema_type_annotation",
    "torch.fx.experimental.symbolic_shapes",
    "torch.fx.passes.backends.cudagraphs",
    "torch.fx.passes.infra.partitioner",
    "torch.fx.passes.utils.fuser_utils",
}


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
            "Block",
            "BoolType",
            "BufferDict",
            "StorageBase",
            "CallStack",
            "Capsule",
            "ClassType",
            "clear_autocast_cache",
            "Code",
            "CompilationUnit",
            "CompleteArgumentSpec",
            "ComplexType",
            "ConcreteModuleType",
            "ConcreteModuleTypeBuilder",
            "CONV_BN_FUSION",
            "cpp",
            "CudaBFloat16TensorBase",
            "CudaBFloat16TensorBase",
            "CudaBoolTensorBase",
            "CudaBoolTensorBase",
            "CudaByteTensorBase",
            "CudaByteTensorBase",
            "CudaCharTensorBase",
            "CudaCharTensorBase",
            "CudaComplexDoubleTensorBase",
            "CudaComplexDoubleTensorBase",
            "CudaComplexFloatTensorBase",
            "CudaComplexFloatTensorBase",
            "CudaDoubleTensorBase",
            "CudaDoubleTensorBase",
            "CudaFloatTensorBase",
            "CudaHalfTensorBase",
            "CudaIntTensorBase",
            "CudaIntTensorBase",
            "CudaLongTensorBase",
            "CudaLongTensorBase",
            "CudaShortTensorBase",
            "CudaShortTensorBase",
            "DeepCopyMemoTable",
            "default_generator",
            "DeserializationStorageContext",
            "device",
            "DeviceObjType",
            "DictType",
            "DisableTorchFunction",
            "DispatchKey",
            "DispatchKeySet",
            "dtype",
            "EnumType",
            "ErrorReport",
            "ExcludeDispatchKeyGuard",
            "ExecutionPlan",
            "FatalError",
            "FileCheck",
            "finfo",
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
            "has_cuda",
            "has_cudnn",
            "has_lapack",
            "has_mkl",
            "has_mkldnn",
            "has_mps",
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
            "IntType",
            "SymFloatType",
            "SymIntType",
            "IODescriptor",
            "is_anomaly_enabled",
            "is_anomaly_check_nan_enabled",
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
            "qscheme",
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
            "Size",
            "StaticModule",
            "Stream",
            "StreamObjType",
            "StringType",
            "SUM",
            "SymFloat",
            "SymInt",
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
            "has_mps",
            "has_openmp",
            "iinfo",
            "import_ir_module",
            "import_ir_module_from_buffer",
            "init_num_threads",
            "is_anomaly_enabled",
            "is_anomaly_check_nan_enabled",
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
            "VULKAN_AUTOMATIC_GPU_TRANSFER",
            "wait",
            "Tag",
        }
        torch_C_bindings = {elem for elem in dir(torch._C) if not elem.startswith("_")}

        # Check that the torch._C bindings are all in the allowlist. Since
        # bindings can change based on how PyTorch was compiled (e.g. with/without
        # CUDA), the two may not be an exact match but the bindings should be
        # a subset of the allowlist.
        difference = torch_C_bindings.difference(torch_C_allowlist_superset)
        msg = f"torch._C had bindings that are not present in the allowlist:\n{difference}"
        self.assertTrue(torch_C_bindings.issubset(torch_C_allowlist_superset), msg)

    # AttributeError: module 'torch.distributed' has no attribute '_shard'
    @unittest.skipIf(IS_WINDOWS, "Distributed Attribute Error")
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
        with open(os.path.join(os.path.dirname(__file__), 'allowlist_for_publicAPI.json')) as json_file:
            # no new entries should be added to this allow_dict.
            # New APIs must follow the public API guidelines.
            allow_dict = json.load(json_file)
            # Because we want minimal modifications to the `allowlist_for_publicAPI.json`,
            # we are adding the entries for the migrated modules here from the original
            # locations.
            for modname in allow_dict["being_migrated"]:
                if modname in allow_dict:
                    allow_dict[allow_dict["being_migrated"][modname]] = allow_dict[modname]

        def test_module(modname):
            split_strs = modname.split('.')
            mod = sys.modules.get(modname)
            for elem in split_strs:
                if elem.startswith("_"):
                    return

            # verifies that each public API has the correct module name and naming semantics
            def check_one_element(elem, modname, mod, *, is_public, is_all):
                obj = getattr(mod, elem)
                if not (isinstance(obj, Callable) or inspect.isclass(obj)):
                    return
                elem_module = getattr(obj, '__module__', None)
                # Only used for nice error message below
                why_not_looks_public = ""
                if elem_module is None:
                    why_not_looks_public = "because it does not have a `__module__` attribute"
                # If a module is being migrated from foo.a to bar.a (that is entry {"foo": "bar"}),
                # the module's starting package would be referred to as the new location even
                # if there is a "from foo import a" inside the "bar.py".
                modname = allow_dict["being_migrated"].get(modname, modname)
                elem_modname_starts_with_mod = elem_module is not None and \
                    elem_module.startswith(modname) and \
                    '._' not in elem_module
                if not why_not_looks_public and not elem_modname_starts_with_mod:
                    why_not_looks_public = f"because its `__module__` attribute (`{elem_module}`) is not within the " \
                        f"torch library or does not start with the submodule where it is defined (`{modname}`)"
                # elem's name must NOT begin with an `_` and it's module name
                # SHOULD start with it's current module since it's a public API
                looks_public = not elem.startswith('_') and elem_modname_starts_with_mod
                if not why_not_looks_public and not looks_public:
                    why_not_looks_public = f"because it starts with `_` (`{elem}`)"

                if is_public != looks_public:
                    if modname in NOT_IMPORTED_WHEN_TEST_WRITTEN:
                        return
                    if modname in allow_dict and elem in allow_dict[modname]:
                        return

                    if is_public:
                        why_is_public = f"it is inside the module's (`{modname}`) `__all__`" if is_all else \
                            "it is an attribute that does not start with `_` on a module that " \
                            "does not have `__all__` defined"
                        fix_is_public = f"remove it from the modules's (`{modname}`) `__all__`" if is_all else \
                            f"either define a `__all__` for `{modname}` or add a `_` at the beginning of the name"
                    else:
                        assert is_all
                        why_is_public = f"it is not inside the module's (`{modname}`) `__all__`"
                        fix_is_public = f"add it from the modules's (`{modname}`) `__all__`"

                    if looks_public:
                        why_looks_public = "it does look public because it follows the rules from the doc above " \
                            "(does not start with `_` and has a proper `__module__`)."
                        fix_looks_public = "make its name start with `_`"
                    else:
                        why_looks_public = why_not_looks_public
                        if not elem_modname_starts_with_mod:
                            fix_looks_public = "make sure the `__module__` is properly set and points to a submodule "\
                                f"of `{modname}`"
                        else:
                            fix_looks_public = "remove the `_` at the beginning of the name"

                    failure_list.append(f"# {modname}.{elem}:")
                    is_public_str = "" if is_public else " NOT"
                    failure_list.append(f"  - Is{is_public_str} public: {why_is_public}")
                    looks_public_str = "" if looks_public else " NOT"
                    failure_list.append(f"  - Does{looks_public_str} look public: {why_looks_public}")
                    # Swap the str below to avoid having to create the NOT again
                    failure_list.append("  - You can do either of these two things to fix this problem:")
                    failure_list.append(f"    - To make it{looks_public_str} public: {fix_is_public}")
                    failure_list.append(f"    - To make it{is_public_str} look public: {fix_looks_public}")


            if hasattr(mod, '__all__'):
                public_api = mod.__all__
                all_api = dir(mod)
                for elem in all_api:
                    check_one_element(elem, modname, mod, is_public=elem in public_api, is_all=True)

            else:
                all_api = dir(mod)
                for elem in all_api:
                    if not elem.startswith('_'):
                        check_one_element(elem, modname, mod, is_public=True, is_all=False)
        for _, modname, ispkg in pkgutil.walk_packages(path=torch.__path__, prefix=torch.__name__ + '.'):
            test_module(modname)

        test_module('torch')

        msg = "All the APIs below do not meet our guidelines for public API from " \
              "https://github.com/pytorch/pytorch/wiki/Public-API-definition-and-documentation.\n"
        msg += "Make sure that everything that is public is expected (in particular that the module " \
            "has a properly populated `__all__` attribute) and that everything that is supposed to be public " \
            "does look public (it does not start with `_` and has a `__module__` that is properly populated)."
        msg += "\n\nFull list:\n"
        msg += "\n".join(map(str, failure_list))

        # empty lists are considered false in python
        self.assertTrue(not failure_list, msg)

if __name__ == '__main__':
    run_tests()
