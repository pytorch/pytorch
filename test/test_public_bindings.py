from torch.testing._internal.common_utils import run_tests, IS_IN_CI

import torch
import unittest
import urllib.request
import io
import zipfile
import pathlib
import sys
import textwrap
import subprocess
import os


from typing import Set


PR_BODY = os.getenv("GITHUB_PR_BODY", None)


def parse_api_list(api_list: str) -> Set[str]:
    return set(api_list.strip().split("\n"))


def fetch_master_python_apis() -> Set[str]:
    APIS_URL = "https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch/ci/apis.zip"

    with urllib.request.urlopen(APIS_URL) as f:
        zip_buf = io.BytesIO(f.read())

    with zipfile.ZipFile(zip_buf) as f:
        return parse_api_list(f.read("apis.log").decode())


def fetch_current_python_apis() -> Set[str]:
    repo_dir = pathlib.Path(__file__).resolve().parent.parent
    list_apis_script = repo_dir / "scripts" / "list_apis.py"
    cmd = [sys.executable, str(list_apis_script), "--module", "torch", "--public"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if proc.returncode != 0:
        raise RuntimeError(f"Failed to generate APIs from local build: {proc.stderr}")

    return parse_api_list(proc.stdout.decode())


class TestPublicBindings(unittest.TestCase):
    @unittest.skipIf(not IS_IN_CI or PR_BODY is None, "API bindings are only tested on PRs")
    def test_no_new_python_bindings(self):
        """
        This implements https://github.com/pytorch/pytorch/issues/58617, which
        will fail if the PR this runs on introduces new public APIs. A public
        API is considered anything that is not a child of a system module (e.g.
        'sys', 'typing', etc.) that is a child of the 'torch' module (e.g.
        'torch.ones', 'torch.jit.script', etc.). Unlike the C bindings there are
        several thousand Python API bindings, so we rely on a list stored in S3
        and updated by a job on master in .github/templates/linux_ci_workflow.yml.j2

        If this test is run on master or can't reach S3, it is a no-op.
        """
        try:
            master_public_bindings = fetch_master_python_apis()
            current_public_bindings = fetch_current_python_apis()
        except Exception as e:
            print(f"failed to check public Python APIs: {e}")
            return

        difference = master_public_bindings.symmetric_difference(current_public_bindings)
        msg = textwrap.dedent(f"""
            New public API(s) were added: {', '.join(difference)}

            To silence this error, add a release notes section to your PR body:

                ### Release Notes
                * the new feature""")
        self.assertTrue(difference != set() and "### Release Notes" not in PR_BODY, msg)

    def test_no_new_c_bindings(self):
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
        torch_C_allowlist = {
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
            "CudaBoolStorageBase",
            "CudaBoolTensorBase",
            "CudaByteStorageBase",
            "CudaByteTensorBase",
            "CudaCharStorageBase",
            "CudaCharTensorBase",
            "CudaComplexDoubleStorageBase",
            "CudaComplexDoubleTensorBase",
            "CudaComplexFloatStorageBase",
            "CudaComplexFloatTensorBase",
            "CudaDoubleStorageBase",
            "CudaDoubleTensorBase",
            "CudaFloatStorageBase",
            "CudaFloatTensorBase",
            "CudaHalfStorageBase",
            "CudaHalfTensorBase",
            "CudaIntStorageBase",
            "CudaIntTensorBase",
            "CudaLongStorageBase",
            "CudaLongTensorBase",
            "CudaShortStorageBase",
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
            "QUInt8StorageBase",
            "read_vitals",
            "REMOVE_DROPOUT",
            "RRefType",
            "ScriptClass",
            "ScriptClassFunction",
            "ScriptDict",
            "ScriptDictIterator",
            "ScriptDictKeyIterator",
            "ScriptFunction",
            "ScriptMethod",
            "ScriptModule",
            "ScriptModuleSerializer",
            "ScriptObject",
            "ScriptObjectProperty",
            "SerializationStorageContext",
            "set_anomaly_enabled",
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
            "Use",
            "Value",
            "vitals_enabled",
            "wait",
        }
        torch_C_bindings = {elem for elem in dir(torch._C) if not elem.startswith("_")}

        # Check that both sets above have the same elements as each other.
        self.assertSetEqual(torch_C_bindings, torch_C_allowlist)


if __name__ == '__main__':
    run_tests()
