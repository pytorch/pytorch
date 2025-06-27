# Owner(s): ["module: cpp-extensions"]

import glob
import locale
import os
import random
import re
import shutil
import string
import subprocess
import sys
import tempfile
import unittest
import warnings

import torch
import torch.backends.cudnn
import torch.multiprocessing as mp
import torch.testing._internal.common_utils as common
import torch.utils.cpp_extension
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_CUDNN
from torch.testing._internal.common_utils import gradcheck, TEST_XPU
from torch.utils.cpp_extension import (
    _get_cuda_arch_flags,
    _TORCH_PATH,
    check_compiler_is_gcc,
    CUDA_HOME,
    get_cxx_compiler,
    remove_extension_h_precompiler_headers,
    ROCM_HOME,
)


# define TEST_ROCM before changing TEST_CUDA
TEST_ROCM = TEST_CUDA and torch.version.hip is not None and ROCM_HOME is not None
TEST_CUDA = TEST_CUDA and CUDA_HOME is not None
TEST_MPS = torch.backends.mps.is_available()
IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")


# There's only one test that runs gradcheck, run slow mode manually
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCppExtensionJIT(common.TestCase):
    """Tests just-in-time cpp extensions.
    Don't confuse this with the PyTorch JIT (aka TorchScript).
    """

    def setUp(self):
        super().setUp()
        # cpp extensions use relative paths. Those paths are relative to
        # this file, so we'll change the working directory temporarily
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def tearDown(self):
        super().tearDown()
        # return the working directory (see setUp)
        os.chdir(self.old_working_dir)

    @classmethod
    def setUpClass(cls):
        torch.testing._internal.common_utils.remove_cpp_extensions_build_root()

    @classmethod
    def tearDownClass(cls):
        torch.testing._internal.common_utils.remove_cpp_extensions_build_root()

    def test_jit_compile_extension(self):
        module = torch.utils.cpp_extension.load(
            name="jit_extension",
            sources=[
                "cpp_extensions/jit_extension.cpp",
                "cpp_extensions/jit_extension2.cpp",
            ],
            extra_include_paths=[
                "cpp_extensions",
                "path / with spaces in it",
                "path with quote'",
            ],
            extra_cflags=["-g"],
            verbose=True,
        )
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        z = module.tanh_add(x, y)
        self.assertEqual(z, x.tanh() + y.tanh())

        # Checking we can call a method defined not in the main C++ file.
        z = module.exp_add(x, y)
        self.assertEqual(z, x.exp() + y.exp())

        # Checking we can use this JIT-compiled class.
        doubler = module.Doubler(2, 2)
        self.assertIsNone(doubler.get().grad)
        self.assertEqual(doubler.get().sum(), 4)
        self.assertEqual(doubler.forward().sum(), 8)

    @unittest.skipIf(not (TEST_CUDA or TEST_ROCM), "CUDA not found")
    def test_jit_cuda_extension(self):
        # NOTE: The name of the extension must equal the name of the module.
        module = torch.utils.cpp_extension.load(
            name="torch_test_cuda_extension",
            sources=[
                "cpp_extensions/cuda_extension.cpp",
                "cpp_extensions/cuda_extension.cu",
            ],
            extra_cuda_cflags=["-O2"],
            verbose=True,
            keep_intermediates=False,
        )

        x = torch.zeros(100, device="cuda", dtype=torch.float32)
        y = torch.zeros(100, device="cuda", dtype=torch.float32)

        z = module.sigmoid_add(x, y).cpu()

        # 2 * sigmoid(0) = 2 * 0.5 = 1
        self.assertEqual(z, torch.ones_like(z))

    def _test_jit_xpu_extension(self, extra_sycl_cflags):
        # randomizing extension name and names of extension methods
        # for the case when we test building few extensions in a row
        # using this function
        rand = "".join(random.sample(string.ascii_letters, 5))
        name = f"torch_test_xpu_extension_{rand}"
        temp_dir = tempfile.mkdtemp()
        try:
            with open("cpp_extensions/xpu_extension.sycl") as f:
                text = f.read()
                for fn in ["sigmoid_add", "SigmoidAddKernel"]:
                    text = text.replace(fn, f"{fn}_{rand}")

            sycl_file = f"{temp_dir}/xpu_extension.sycl"
            with open(sycl_file, "w") as f:
                f.write(text)

            module = torch.utils.cpp_extension.load(
                name=name,
                sources=[sycl_file],
                extra_sycl_cflags=extra_sycl_cflags,
                verbose=True,
                keep_intermediates=True,
                build_directory=temp_dir,
            )

            x = torch.zeros(100, device="xpu", dtype=torch.float32)
            y = torch.zeros(100, device="xpu", dtype=torch.float32)

            method = f"sigmoid_add_{rand}"
            self.assertTrue(hasattr(module, method))
            z = getattr(module, method)(x, y).cpu()

            # 2 * sigmoid(0) = 2 * 0.5 = 1
            self.assertEqual(z, torch.ones_like(z))
        finally:
            shutil.rmtree(temp_dir)

    @unittest.skipIf(not (TEST_XPU), "XPU not found")
    def test_jit_xpu_extension(self):
        # NOTE: this test can be affected by setting TORCH_XPU_ARCH_LIST
        self._test_jit_xpu_extension(extra_sycl_cflags=[])

    @unittest.skipIf(not (TEST_XPU), "XPU not found")
    def test_jit_xpu_archlists(self):
        # NOTE: in this test we explicitly test few different options
        # for TORCH_XPU_ARCH_LIST. Setting TORCH_XPU_ARCH_LIST in the
        # environment before the test won't affect it.
        cases = [
            {
                # Testing JIT compilation
                "archlist": "",
                "extra_sycl_cflags": [],
            },
            {
                # Testing JIT + AOT (full torch AOT arch list)
                # NOTE: default cpp extension AOT arch list might be reduced
                # from the full list
                "archlist": ",".join(torch.xpu.get_arch_list()),
                "extra_sycl_cflags": [],
            },
            {
                # Testing AOT (full torch AOT arch list)
                # NOTE: default cpp extension AOT arch list might be reduced
                # from the full list
                "archlist": ",".join(torch.xpu.get_arch_list()),
                # below excludes spir64 target responsible for JIT
                "extra_sycl_cflags": ["-fsycl-targets=spir64_gen"],
            },
        ]
        old_envvar = os.environ.get("TORCH_XPU_ARCH_LIST", None)
        try:
            for c in cases:
                os.environ["TORCH_XPU_ARCH_LIST"] = c["archlist"]
                self._test_jit_xpu_extension(extra_sycl_cflags=c["extra_sycl_cflags"])
        finally:
            if old_envvar is None:
                os.environ.pop("TORCH_XPU_ARCH_LIST")
            else:
                os.environ["TORCH_XPU_ARCH_LIST"] = old_envvar

    @unittest.skipIf(not TEST_MPS, "MPS not found")
    def test_mps_extension(self):
        module = torch.utils.cpp_extension.load(
            name="torch_test_mps_extension",
            sources=[
                "cpp_extensions/mps_extension.mm",
            ],
            verbose=True,
            keep_intermediates=False,
        )

        tensor_length = 100000
        x = torch.randn(tensor_length, device="cpu", dtype=torch.float32)
        y = torch.randn(tensor_length, device="cpu", dtype=torch.float32)

        cpu_output = module.get_cpu_add_output(x, y)
        mps_output = module.get_mps_add_output(x.to("mps"), y.to("mps"))

        self.assertEqual(cpu_output, mps_output.to("cpu"))

    def _run_jit_cuda_archflags(self, flags, expected):
        # Compile an extension with given `flags`
        def _check_cuobjdump_output(expected_values, is_ptx=False):
            elf_or_ptx = "--list-ptx" if is_ptx else "--list-elf"
            lib_ext = ".pyd" if IS_WINDOWS else ".so"
            # Note, .extension name may include _v1, _v2, so first find exact name
            ext_filename = glob.glob(
                os.path.join(temp_dir, "cudaext_archflag*" + lib_ext)
            )[0]
            command = ["cuobjdump", elf_or_ptx, ext_filename]
            p = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            output, err = p.communicate()
            output = output.decode("ascii")
            err = err.decode("ascii")

            if not p.returncode == 0 or not err == "":
                raise AssertionError(
                    f"Flags: {flags}\nReturncode: {p.returncode}\nStderr: {err}\n"
                    f"Output: {output} "
                )

            actual_arches = sorted(re.findall(r"sm_\d+", output))
            expected_arches = sorted(
                ["sm_" + xx.replace("121", "120") for xx in expected_values]
            )
            self.assertEqual(
                actual_arches,
                expected_arches,
                msg=f"Flags: {flags},  Actual: {actual_arches},  Expected: {expected_arches}\n"
                f"Stderr: {err}\nOutput: {output}",
            )

        temp_dir = tempfile.mkdtemp()
        old_envvar = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
        try:
            os.environ["TORCH_CUDA_ARCH_LIST"] = flags

            params = {
                "name": "cudaext_archflags",
                "sources": [
                    "cpp_extensions/cuda_extension.cpp",
                    "cpp_extensions/cuda_extension.cu",
                ],
                "extra_cuda_cflags": ["-O2"],
                "verbose": True,
                "build_directory": temp_dir,
            }

            if IS_WINDOWS:
                p = mp.Process(target=torch.utils.cpp_extension.load, kwargs=params)

                # Compile and load the test CUDA arch in a different Python process to avoid
                # polluting the current one and causes test_jit_cuda_extension to fail on
                # Windows. There is no clear way to unload a module after it has been imported
                # and torch.utils.cpp_extension.load builds and loads the module in one go.
                # See https://github.com/pytorch/pytorch/issues/61655 for more details
                p.start()
                p.join()
            else:
                torch.utils.cpp_extension.load(**params)

            # Expected output for --list-elf:
            #   ELF file    1: cudaext_archflags.1.sm_61.cubin
            #   ELF file    2: cudaext_archflags.2.sm_52.cubin
            _check_cuobjdump_output(expected[0])
            if expected[1] is not None:
                # Expected output for --list-ptx:
                #   PTX file    1: cudaext_archflags.1.sm_61.ptx
                _check_cuobjdump_output(expected[1], is_ptx=True)
        finally:
            if IS_WINDOWS:
                # rmtree returns permission error: [WinError 5] Access is denied
                # on Windows, this is a word-around
                subprocess.run(["rm", "-rf", temp_dir], stdout=subprocess.PIPE)
            else:
                shutil.rmtree(temp_dir)

            if old_envvar is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST")
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = old_envvar

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    @unittest.skipIf(TEST_ROCM, "disabled on rocm")
    def test_jit_cuda_archflags(self):
        # Test a number of combinations:
        #   - the default for the machine we're testing on
        #   - Separators, can be ';' (most common) or ' '
        #   - Architecture names
        #   - With/without '+PTX'

        n = torch.cuda.device_count()
        capabilities = {torch.cuda.get_device_capability(i) for i in range(n)}
        # expected values is length-2 tuple: (list of ELF, list of PTX)
        # note: there should not be more than one PTX value
        archflags = {
            "": (
                [f"{capability[0]}{capability[1]}" for capability in capabilities],
                None,
            ),
            "Maxwell+Tegra;6.1": (["53", "61"], None),
            "Volta": (["70"], ["70"]),
        }
        archflags["7.5+PTX"] = (["75"], ["75"])
        archflags["5.0;6.0+PTX;7.0;7.5"] = (["50", "60", "70", "75"], ["60"])
        if int(torch.version.cuda.split(".")[0]) < 12:
            # CUDA 12 drops compute capability < 5.0
            archflags["Pascal 3.5"] = (["35", "60", "61"], None)

        for flags, expected in archflags.items():
            try:
                self._run_jit_cuda_archflags(flags, expected)
            except RuntimeError as e:
                # Using the device default (empty flags) may fail if the device is newer than the CUDA compiler
                # This raises a RuntimeError with a specific message which we explicitly ignore here
                if not flags and "Error building" in str(e):
                    pass
                else:
                    raise
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                # Ignore any error, e.g. unsupported PTX code on current device
                # to avoid errors from here leaking into other tests
                pass

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_cuda_arch_flags_non_default_gencode(self):
        user_arch_flags = ["-gencode=arch=compute_86,code=sm_86"]
        result = _get_cuda_arch_flags(user_arch_flags)

        self.assertEqual(
            len(result),
            0,
            f"User arch flags should prevent default generation. "
            f"Expected: [], Got: {result}",
        )

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_cuda_arch_flags_default_gencode(self):
        default_flags = _get_cuda_arch_flags()
        self.assertGreater(
            len(default_flags), 0, "No args should generate default flags"
        )

        non_arch_flags = _get_cuda_arch_flags(["-O2", "--use-fast-math"])
        self.assertGreater(
            len(non_arch_flags), 0, "Non-arch flags should still generate defaults"
        )

        empty_flags = _get_cuda_arch_flags([])
        self.assertGreater(
            len(empty_flags), 0, "Empty list should generate default flags"
        )

    @unittest.skipIf(not TEST_CUDNN, "CuDNN not found")
    @unittest.skipIf(TEST_ROCM, "Not supported on ROCm")
    def test_jit_cudnn_extension(self):
        # implementation of CuDNN ReLU
        if IS_WINDOWS:
            extra_ldflags = ["cudnn.lib"]
        else:
            extra_ldflags = ["-lcudnn"]
        module = torch.utils.cpp_extension.load(
            name="torch_test_cudnn_extension",
            sources=["cpp_extensions/cudnn_extension.cpp"],
            extra_ldflags=extra_ldflags,
            verbose=True,
            with_cuda=True,
        )

        x = torch.randn(100, device="cuda", dtype=torch.float32)
        y = torch.zeros(100, device="cuda", dtype=torch.float32)
        module.cudnn_relu(x, y)  # y=relu(x)
        self.assertEqual(torch.nn.functional.relu(x), y)
        with self.assertRaisesRegex(RuntimeError, "same size"):
            y_incorrect = torch.zeros(20, device="cuda", dtype=torch.float32)
            module.cudnn_relu(x, y_incorrect)

    def test_inline_jit_compile_extension_with_functions_as_list(self):
        cpp_source = """
        torch::Tensor tanh_add(torch::Tensor x, torch::Tensor y) {
          return x.tanh() + y.tanh();
        }
        """

        module = torch.utils.cpp_extension.load_inline(
            name="inline_jit_extension_with_functions_list",
            cpp_sources=cpp_source,
            functions="tanh_add",
            verbose=True,
        )

        self.assertEqual(module.tanh_add.__doc__.split("\n")[2], "tanh_add")

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        z = module.tanh_add(x, y)
        self.assertEqual(z, x.tanh() + y.tanh())

    def test_inline_jit_compile_extension_with_functions_as_dict(self):
        cpp_source = """
        torch::Tensor tanh_add(torch::Tensor x, torch::Tensor y) {
          return x.tanh() + y.tanh();
        }
        """

        module = torch.utils.cpp_extension.load_inline(
            name="inline_jit_extension_with_functions_dict",
            cpp_sources=cpp_source,
            functions={"tanh_add": "Tanh and then sum :D"},
            verbose=True,
        )

        self.assertEqual(module.tanh_add.__doc__.split("\n")[2], "Tanh and then sum :D")

    def test_inline_jit_compile_extension_multiple_sources_and_no_functions(self):
        cpp_source1 = """
        torch::Tensor sin_add(torch::Tensor x, torch::Tensor y) {
          return x.sin() + y.sin();
        }
        """

        cpp_source2 = """
        #include <torch/extension.h>
        torch::Tensor sin_add(torch::Tensor x, torch::Tensor y);
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
          m.def("sin_add", &sin_add, "sin(x) + sin(y)");
        }
        """

        module = torch.utils.cpp_extension.load_inline(
            name="inline_jit_extension",
            cpp_sources=[cpp_source1, cpp_source2],
            verbose=True,
        )

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        z = module.sin_add(x, y)
        self.assertEqual(z, x.sin() + y.sin())

    @unittest.skip("Temporarily disabled")
    @unittest.skipIf(not (TEST_CUDA or TEST_ROCM), "CUDA not found")
    def test_inline_jit_compile_extension_cuda(self):
        cuda_source = """
        __global__ void cos_add_kernel(
            const float* __restrict__ x,
            const float* __restrict__ y,
            float* __restrict__ output,
            const int size) {
          const auto index = blockIdx.x * blockDim.x + threadIdx.x;
          if (index < size) {
            output[index] = __cosf(x[index]) + __cosf(y[index]);
          }
        }

        torch::Tensor cos_add(torch::Tensor x, torch::Tensor y) {
          auto output = torch::zeros_like(x);
          const int threads = 1024;
          const int blocks = (output.numel() + threads - 1) / threads;
          cos_add_kernel<<<blocks, threads>>>(x.data<float>(), y.data<float>(), output.data<float>(), output.numel());
          return output;
        }
        """

        # Here, the C++ source need only declare the function signature.
        cpp_source = "torch::Tensor cos_add(torch::Tensor x, torch::Tensor y);"

        module = torch.utils.cpp_extension.load_inline(
            name="inline_jit_extension_cuda",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=["cos_add"],
            verbose=True,
        )

        self.assertEqual(module.cos_add.__doc__.split("\n")[2], "cos_add")

        x = torch.randn(4, 4, device="cuda", dtype=torch.float32)
        y = torch.randn(4, 4, device="cuda", dtype=torch.float32)

        z = module.cos_add(x, y)
        self.assertEqual(z, x.cos() + y.cos())

    @unittest.skip("Temporarily disabled")
    @unittest.skipIf(not (TEST_CUDA or TEST_ROCM), "CUDA not found")
    def test_inline_jit_compile_custom_op_cuda(self):
        cuda_source = """
        __global__ void cos_add_kernel(
            const float* __restrict__ x,
            const float* __restrict__ y,
            float* __restrict__ output,
            const int size) {
          const auto index = blockIdx.x * blockDim.x + threadIdx.x;
          if (index < size) {
            output[index] = __cosf(x[index]) + __cosf(y[index]);
          }
        }

        torch::Tensor cos_add(torch::Tensor x, torch::Tensor y) {
          auto output = torch::zeros_like(x);
          const int threads = 1024;
          const int blocks = (output.numel() + threads - 1) / threads;
          cos_add_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), output.data_ptr<float>(), output.numel());
          return output;
        }
        """

        # Here, the C++ source need only declare the function signature.
        cpp_source = """
           #include <torch/library.h>
           torch::Tensor cos_add(torch::Tensor x, torch::Tensor y);

           TORCH_LIBRARY(inline_jit_extension_custom_op_cuda, m) {
             m.def("cos_add", cos_add);
           }
        """

        torch.utils.cpp_extension.load_inline(
            name="inline_jit_extension_custom_op_cuda",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            verbose=True,
            is_python_module=False,
        )

        x = torch.randn(4, 4, device="cuda", dtype=torch.float32)
        y = torch.randn(4, 4, device="cuda", dtype=torch.float32)

        z = torch.ops.inline_jit_extension_custom_op_cuda.cos_add(x, y)
        self.assertEqual(z, x.cos() + y.cos())

    @unittest.skipIf(not TEST_XPU, "XPU not found")
    def test_inline_jit_compile_extension_xpu(self):
        sycl_source = """
        #include <c10/xpu/XPUStream.h>

        class CosAddKernel {
        public:
          void operator()(const sycl::nd_item<3> &item_ct1) const {
            const int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                              item_ct1.get_local_id(2);
            if (index < size) {
              output[index] = cosf(x[index]) + cosf(y[index]);
            }
          }
          CosAddKernel(const float* _x, const float* _y, float* _output, int _size):
            x(_x),
            y(_y),
            output(_output),
            size(_size)
          {}
        private:
          const float* x;
          const float* y;
          float* output;
          int size;
        };

        void cos_add_kernel(
            const float* x,
            const float* y,
            float* output,
            int size) {
          CosAddKernel krn(x, y, output, size);
          const int threads = 1024;
          const int blocks = (size + threads - 1) / threads;

          sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
          queue.submit([&](sycl::handler &cgh) {
              cgh.parallel_for<CosAddKernel>(
                  sycl::nd_range<3>(
                      sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads),
                      sycl::range<3>(1, 1, threads)),
              krn);
          });
        }

        torch::Tensor cos_add(torch::Tensor x, torch::Tensor y) {
          auto output = torch::zeros_like(x);
          const int threads = 1024;
          const int blocks = (output.numel() + threads - 1) / threads;
          cos_add_kernel(x.data_ptr<float>(), y.data_ptr<float>(), output.data_ptr<float>(), output.numel());
          return output;
        }
        """

        # Here, the C++ source need only declare the function signature.
        cpp_source = "torch::Tensor cos_add(torch::Tensor x, torch::Tensor y);"

        module = torch.utils.cpp_extension.load_inline(
            name="inline_jit_extension_xpu",
            cpp_sources=cpp_source,
            sycl_sources=sycl_source,
            functions=["cos_add"],
            verbose=True,
        )

        self.assertEqual(module.cos_add.__doc__.split("\n")[2], "cos_add")

        x = torch.randn(4, 4, device="xpu", dtype=torch.float32)
        y = torch.randn(4, 4, device="xpu", dtype=torch.float32)

        z = module.cos_add(x, y)
        self.assertEqual(z, x.cos() + y.cos())

    def test_inline_jit_compile_extension_throws_when_functions_is_bad(self):
        with self.assertRaises(ValueError):
            torch.utils.cpp_extension.load_inline(
                name="invalid_jit_extension", cpp_sources="", functions=5
            )

    def test_lenient_flag_handling_in_jit_extensions(self):
        cpp_source = """
        torch::Tensor tanh_add(torch::Tensor x, torch::Tensor y) {
          return x.tanh() + y.tanh();
        }
        """

        module = torch.utils.cpp_extension.load_inline(
            name="lenient_flag_handling_extension",
            cpp_sources=cpp_source,
            functions="tanh_add",
            extra_cflags=["-g\n\n", "-O0 -Wall"],
            extra_include_paths=["       cpp_extensions\n"],
            verbose=True,
        )

        x = torch.zeros(100, dtype=torch.float32)
        y = torch.zeros(100, dtype=torch.float32)
        z = module.tanh_add(x, y).cpu()
        self.assertEqual(z, x.tanh() + y.tanh())

    @unittest.skip("Temporarily disabled")
    @unittest.skipIf(not (TEST_CUDA or TEST_ROCM), "CUDA not found")
    def test_half_support(self):
        """
        Checks for an issue with operator< ambiguity for half when certain
        THC headers are included.

        See https://github.com/pytorch/pytorch/pull/10301#issuecomment-416773333
        for the corresponding issue.
        """
        cuda_source = """
        template<typename T, typename U>
        __global__ void half_test_kernel(const T* input, U* output) {
            if (input[0] < input[1] || input[0] >= input[1]) {
                output[0] = 123;
            }
        }

        torch::Tensor half_test(torch::Tensor input) {
            auto output = torch::empty(1, input.options().dtype(torch::kFloat));
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "half_test", [&] {
                half_test_kernel<scalar_t><<<1, 1>>>(
                    input.data<scalar_t>(),
                    output.data<float>());
            });
            return output;
        }
        """

        module = torch.utils.cpp_extension.load_inline(
            name="half_test_extension",
            cpp_sources="torch::Tensor half_test(torch::Tensor input);",
            cuda_sources=cuda_source,
            functions=["half_test"],
            verbose=True,
        )

        x = torch.randn(3, device="cuda", dtype=torch.half)
        result = module.half_test(x)
        self.assertEqual(result[0], 123)

    def test_reload_jit_extension(self):
        def compile(code):
            return torch.utils.cpp_extension.load_inline(
                name="reloaded_jit_extension",
                cpp_sources=code,
                functions="f",
                verbose=True,
            )

        module = compile("int f() { return 123; }")
        self.assertEqual(module.f(), 123)

        module = compile("int f() { return 456; }")
        self.assertEqual(module.f(), 456)
        module = compile("int f() { return 456; }")
        self.assertEqual(module.f(), 456)

        module = compile("int f() { return 789; }")
        self.assertEqual(module.f(), 789)

    @unittest.skipIf(
        "utf" not in locale.getlocale()[1].lower(), "Only test in UTF-8 locale"
    )
    def test_load_with_non_platform_default_encoding(self):
        # Assume the code is saved in UTF-8, but the locale is set to a different encoding.
        # You might encounter decoding errors in ExtensionVersioner.
        # But this case is quite hard to cover because CI environments may not in non-latin locale.
        # So the following code just test source file in gbk and locale in utf-8.

        cpp_source = """
        #include <torch/extension.h>

        // Non-latin1 character test: 字符.
        // It will cause utf-8 decoding error.

        int f() { return 123; }
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("f", &f, "f");
        }
        """

        build_dir = tempfile.mkdtemp()
        src_path = os.path.join(build_dir, "main.cpp")

        with open(src_path, encoding="gbk", mode="w") as f:
            f.write(cpp_source)

        module = torch.utils.cpp_extension.load(
            name="non_default_encoding",
            sources=src_path,
            verbose=True,
        )
        self.assertEqual(module.f(), 123)

    def test_cpp_frontend_module_has_same_output_as_python(self, dtype=torch.double):
        extension = torch.utils.cpp_extension.load(
            name="cpp_frontend_extension",
            sources="cpp_extensions/cpp_frontend_extension.cpp",
            verbose=True,
        )

        input = torch.randn(2, 5, dtype=dtype)
        cpp_linear = extension.Net(5, 2)
        cpp_linear.to(dtype)
        python_linear = torch.nn.Linear(5, 2).to(dtype)

        # First make sure they have the same parameters
        cpp_parameters = dict(cpp_linear.named_parameters())
        with torch.no_grad():
            python_linear.weight.copy_(cpp_parameters["fc.weight"])
            python_linear.bias.copy_(cpp_parameters["fc.bias"])

        cpp_output = cpp_linear.forward(input)
        python_output = python_linear(input)
        self.assertEqual(cpp_output, python_output)

        cpp_output.sum().backward()
        python_output.sum().backward()

        for p in cpp_linear.parameters():
            self.assertFalse(p.grad is None)

        self.assertEqual(cpp_parameters["fc.weight"].grad, python_linear.weight.grad)
        self.assertEqual(cpp_parameters["fc.bias"].grad, python_linear.bias.grad)

    def test_cpp_frontend_module_python_inter_op(self):
        extension = torch.utils.cpp_extension.load(
            name="cpp_frontend_extension",
            sources="cpp_extensions/cpp_frontend_extension.cpp",
            verbose=True,
        )

        # Create a torch.nn.Module which uses the C++ module as a submodule.
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = torch.nn.Parameter(torch.tensor(1.0))
                self.net = extension.Net(3, 5)

            def forward(self, input):
                return self.net.forward(input) + self.x

        net = extension.Net(5, 2)
        net.double()
        net.to(torch.get_default_dtype())
        self.assertEqual(str(net), "Net")

        # Further embed the torch.nn.Module into a Sequential, and also add the
        # C++ module as an element of the Sequential.
        sequential = torch.nn.Sequential(M(), torch.nn.Tanh(), net, torch.nn.Sigmoid())

        input = torch.randn(2, 3)
        # Try calling the module!
        output = sequential.forward(input)
        # The call operator is bound to forward too.
        self.assertEqual(output, sequential(input))
        self.assertEqual(list(output.shape), [2, 2])

        # Do changes on the module hierarchy.
        old_dtype = torch.get_default_dtype()
        sequential.to(torch.float64)
        sequential.to(torch.float32)
        sequential.to(old_dtype)
        self.assertEqual(sequential[2].parameters()[0].dtype, old_dtype)

        # Make sure we can access these methods recursively.
        self.assertEqual(
            len(list(sequential.parameters())), len(net.parameters()) * 2 + 1
        )
        self.assertEqual(
            len(list(sequential.named_parameters())),
            len(net.named_parameters()) * 2 + 1,
        )
        self.assertEqual(len(list(sequential.buffers())), len(net.buffers()) * 2)
        self.assertEqual(len(list(sequential.modules())), 8)

        # Test clone()
        net2 = net.clone()
        self.assertEqual(len(net.parameters()), len(net2.parameters()))
        self.assertEqual(len(net.buffers()), len(net2.buffers()))
        self.assertEqual(len(net.modules()), len(net2.modules()))

        # Try differentiating through the whole module.
        for parameter in net.parameters():
            self.assertIsNone(parameter.grad)
        output.sum().backward()
        for parameter in net.parameters():
            self.assertFalse(parameter.grad is None)
            self.assertGreater(parameter.grad.sum(), 0)

        # Try calling zero_grad()
        net.zero_grad()
        for p in net.parameters():
            assert p.grad is None, "zero_grad defaults to setting grads to None"

        # Test train(), eval(), training (a property)
        self.assertTrue(net.training)
        net.eval()
        self.assertFalse(net.training)
        net.train()
        self.assertTrue(net.training)
        net.eval()

        # Try calling the additional methods we registered.
        biased_input = torch.randn(4, 5)
        output_before = net.forward(biased_input)
        bias = net.get_bias().clone()
        self.assertEqual(list(bias.shape), [2])
        net.set_bias(bias + 1)
        self.assertEqual(net.get_bias(), bias + 1)
        output_after = net.forward(biased_input)

        self.assertNotEqual(output_before, output_after)

        # Try accessing parameters
        self.assertEqual(len(net.parameters()), 2)
        np = net.named_parameters()
        self.assertEqual(len(np), 2)
        self.assertIn("fc.weight", np)
        self.assertIn("fc.bias", np)

        self.assertEqual(len(net.buffers()), 1)
        nb = net.named_buffers()
        self.assertEqual(len(nb), 1)
        self.assertIn("buf", nb)
        self.assertEqual(nb[0][1], torch.eye(5))

    def test_cpp_frontend_module_has_up_to_date_attributes(self):
        extension = torch.utils.cpp_extension.load(
            name="cpp_frontend_extension",
            sources="cpp_extensions/cpp_frontend_extension.cpp",
            verbose=True,
        )

        net = extension.Net(5, 2)

        self.assertEqual(len(net._parameters), 0)
        net.add_new_parameter("foo", torch.eye(5))
        self.assertEqual(len(net._parameters), 1)

        self.assertEqual(len(net._buffers), 1)
        net.add_new_buffer("bar", torch.eye(5))
        self.assertEqual(len(net._buffers), 2)

        self.assertEqual(len(net._modules), 1)
        net.add_new_submodule("fc2")
        self.assertEqual(len(net._modules), 2)

    @unittest.skipIf(not (TEST_CUDA or TEST_ROCM), "CUDA not found")
    def test_cpp_frontend_module_python_inter_op_with_cuda(self):
        extension = torch.utils.cpp_extension.load(
            name="cpp_frontend_extension",
            sources="cpp_extensions/cpp_frontend_extension.cpp",
            verbose=True,
        )

        net = extension.Net(5, 2)
        for p in net.parameters():
            self.assertTrue(p.device.type == "cpu")
        cpu_parameters = [p.clone() for p in net.parameters()]

        device = torch.device("cuda", 0)
        net.to(device)

        for i, p in enumerate(net.parameters()):
            self.assertTrue(p.device.type == "cuda")
            self.assertTrue(p.device.index == 0)
            self.assertEqual(cpu_parameters[i], p)

        net.cpu()
        net.add_new_parameter("a", torch.eye(5))
        net.add_new_parameter("b", torch.eye(5))
        net.add_new_buffer("c", torch.eye(5))
        net.add_new_buffer("d", torch.eye(5))
        net.add_new_submodule("fc2")
        net.add_new_submodule("fc3")

        for p in net.parameters():
            self.assertTrue(p.device.type == "cpu")

        net.cuda()

        for p in net.parameters():
            self.assertTrue(p.device.type == "cuda")

    def test_returns_shared_library_path_when_is_python_module_is_true(self):
        source = """
        #include <torch/script.h>
        torch::Tensor func(torch::Tensor x) { return x; }
        static torch::RegisterOperators r("test::func", &func);
        """
        torch.utils.cpp_extension.load_inline(
            name="is_python_module",
            cpp_sources=source,
            functions="func",
            verbose=True,
            is_python_module=False,
        )
        self.assertEqual(torch.ops.test.func(torch.eye(5)), torch.eye(5))

    def test_set_default_type_also_changes_aten_default_type(self):
        module = torch.utils.cpp_extension.load_inline(
            name="test_set_default_type",
            cpp_sources="torch::Tensor get() { return torch::empty({}); }",
            functions="get",
            verbose=True,
        )

        initial_default = torch.get_default_dtype()
        try:
            self.assertEqual(module.get().dtype, initial_default)
            torch.set_default_dtype(torch.float64)
            self.assertEqual(module.get().dtype, torch.float64)
            torch.set_default_dtype(torch.float32)
            self.assertEqual(module.get().dtype, torch.float32)
            torch.set_default_dtype(torch.float16)
            self.assertEqual(module.get().dtype, torch.float16)
        finally:
            torch.set_default_dtype(initial_default)

    def test_compilation_error_formatting(self):
        # Test that the missing-semicolon error message has linebreaks in it.
        # This'll fail if the message has been munged into a single line.
        # It's hard to write anything more specific as every compiler has it's own
        # error formatting.
        with self.assertRaises(RuntimeError) as e:
            torch.utils.cpp_extension.load_inline(
                name="test_compilation_error_formatting",
                cpp_sources="int main() { return 0 }",
            )
        pattern = r".*(\\n|\\r).*"
        self.assertNotRegex(str(e), pattern)

    def test_warning(self):
        # Note: the module created from this source will include the py::key_error
        # symbol. But because of visibility and the fact that it lives in a
        # different compilation unit than pybind, this trips up ubsan even though
        # it is fine. "ubsan.supp" thus needs to contain "vptr:warn_mod.so".
        source = """
        // error_type:
        // 0: no error
        // 1: torch::TypeError
        // 2: python_error()
        // 3: py::error_already_set
        at::Tensor foo(at::Tensor x, int error_type) {
            std::ostringstream err_stream;
            err_stream << "Error with "  << x.type();

            TORCH_WARN(err_stream.str());
            if(error_type == 1) {
                throw torch::TypeError(err_stream.str().c_str());
            }
            if(error_type == 2) {
                PyObject* obj = PyTuple_New(-1);
                TORCH_CHECK(!obj);
                // Pretend it was caught in a different thread and restored here
                auto e = python_error();
                e.persist();
                e.restore();
                throw e;
            }
            if(error_type == 3) {
                throw py::key_error(err_stream.str());
            }
            return x.cos();
        }
        """

        # Ensure double type for hard-coded c name below
        t = torch.rand(2).double()
        cpp_tensor_name = r"CPUDoubleType"

        # Without error handling, the warnings cannot be catched
        warn_mod = torch.utils.cpp_extension.load_inline(
            name="warn_mod",
            cpp_sources=[source],
            functions=["foo"],
            with_pytorch_error_handling=False,
        )

        with warnings.catch_warnings(record=True) as w:
            warn_mod.foo(t, 0)
            self.assertEqual(len(w), 0)

            with self.assertRaisesRegex(TypeError, t.type()):
                warn_mod.foo(t, 1)
            self.assertEqual(len(w), 0)

            with self.assertRaisesRegex(
                SystemError, "bad argument to internal function"
            ):
                warn_mod.foo(t, 2)
            self.assertEqual(len(w), 0)

            with self.assertRaisesRegex(KeyError, cpp_tensor_name):
                warn_mod.foo(t, 3)
            self.assertEqual(len(w), 0)

        warn_mod = torch.utils.cpp_extension.load_inline(
            name="warn_mod",
            cpp_sources=[source],
            functions=["foo"],
            with_pytorch_error_handling=True,
        )

        with warnings.catch_warnings(record=True) as w:
            # Catched with no error should be detected
            warn_mod.foo(t, 0)
            self.assertEqual(len(w), 1)

            # Catched with cpp error should also be detected
            with self.assertRaisesRegex(TypeError, t.type()):
                warn_mod.foo(t, 1)
            self.assertEqual(len(w), 2)

            # Catched with python error should also be detected
            with self.assertRaisesRegex(
                SystemError, "bad argument to internal function"
            ):
                warn_mod.foo(t, 2)
            self.assertEqual(len(w), 3)

            # Catched with pybind error should also be detected
            # Note that there is no type name translation for pybind errors
            with self.assertRaisesRegex(KeyError, cpp_tensor_name):
                warn_mod.foo(t, 3)
            self.assertEqual(len(w), 4)

        # Make sure raising warnings are handled properly
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error")

            # No error, the warning should raise
            with self.assertRaisesRegex(UserWarning, t.type()):
                warn_mod.foo(t, 0)
            self.assertEqual(len(w), 0)

            # Another error happened, the warning is ignored
            with self.assertRaisesRegex(TypeError, t.type()):
                warn_mod.foo(t, 1)
            self.assertEqual(len(w), 0)

    def test_autograd_from_cpp(self):
        source = """
        void run_back(at::Tensor x) {
            x.backward({});
        }

        void run_back_no_gil(at::Tensor x) {
            pybind11::gil_scoped_release no_gil;
            x.backward({});
        }
        """

        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gx):
                return gx

        test_backward_deadlock = torch.utils.cpp_extension.load_inline(
            name="test_backward_deadlock",
            cpp_sources=[source],
            functions=["run_back", "run_back_no_gil"],
        )

        # This used to deadlock
        inp = torch.rand(20, requires_grad=True)
        loss = MyFn.apply(inp).sum()
        with self.assertRaisesRegex(
            RuntimeError, "The autograd engine was called while holding the GIL."
        ):
            test_backward_deadlock.run_back(loss)

        inp = torch.rand(20, requires_grad=True)
        loss = MyFn.apply(inp).sum()
        test_backward_deadlock.run_back_no_gil(loss)

    def test_custom_compound_op_autograd(self):
        # Test that a custom compound op (i.e. a custom op that just calls other aten ops)
        # correctly returns gradients of those other ops

        source = """
        #include <torch/library.h>
        torch::Tensor my_add(torch::Tensor x, torch::Tensor y) {
          return x + y;
        }
        TORCH_LIBRARY(my, m) {
            m.def("add", &my_add);
        }
        """

        torch.utils.cpp_extension.load_inline(
            name="is_python_module",
            cpp_sources=source,
            verbose=True,
            is_python_module=False,
        )

        a = torch.randn(5, 5, requires_grad=True)
        b = torch.randn(5, 5, requires_grad=True)

        for fast_mode in (True, False):
            gradcheck(torch.ops.my.add, [a, b], eps=1e-2, fast_mode=fast_mode)

    def test_custom_functorch_error(self):
        # Test that a custom C++ Function raises an error under functorch transforms
        identity_m = torch.utils.cpp_extension.load(
            name="identity",
            sources=["cpp_extensions/identity.cpp"],
        )

        t = torch.randn(3, requires_grad=True)

        msg = r"cannot use C\+\+ torch::autograd::Function with functorch"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.func.vmap(identity_m.identity)(t)

        with self.assertRaisesRegex(RuntimeError, msg):
            torch.func.grad(identity_m.identity)(t)

    def test_gen_extension_h_pch(self):
        if not IS_LINUX:
            return

        source = """
        at::Tensor sin_add(at::Tensor x, at::Tensor y) {
            return x.sin() + y.sin();
        }
        """

        head_file_pch = os.path.join(_TORCH_PATH, "include", "torch", "extension.h.gch")
        head_file_signature = os.path.join(
            _TORCH_PATH, "include", "torch", "extension.h.sign"
        )

        remove_extension_h_precompiler_headers()
        pch_exist = os.path.exists(head_file_pch)
        signature_exist = os.path.exists(head_file_signature)
        self.assertEqual(pch_exist, False)
        self.assertEqual(signature_exist, False)

        torch.utils.cpp_extension.load_inline(
            name="inline_extension_with_pch",
            cpp_sources=[source],
            functions=["sin_add"],
            verbose=True,
            use_pch=True,
        )
        pch_exist = os.path.exists(head_file_pch)
        signature_exist = os.path.exists(head_file_signature)

        compiler = get_cxx_compiler()
        if check_compiler_is_gcc(compiler):
            self.assertEqual(pch_exist, True)
            self.assertEqual(signature_exist, True)

    def test_aoti_torch_call_dispatcher(self):
        source = """
        #include <torch/csrc/inductor/aoti_runtime/utils.h>
        #include <torch/csrc/inductor/aoti_torch/utils.h>
        #include <torch/csrc/inductor/aoti_torch/c/shim.h>
        #include <torch/csrc/stable/library.h>

        using RAIIATH = torch::aot_inductor::RAIIAtenTensorHandle;

        at::Tensor my_abs(at::Tensor x) {
        StableIValue stack[1];
        RAIIATH raii(torch::aot_inductor::new_tensor_handle(std::move(x)));
        stack[0] = from(raii.release());
        aoti_torch_call_dispatcher("aten::abs", "", stack);
        RAIIATH res(to<AtenTensorHandle>(stack[0]));
        return *reinterpret_cast<at::Tensor*>(res.release());
        }

        at::Tensor my_floor(at::Tensor x) {
        StableIValue stack[1];
        RAIIATH raii(torch::aot_inductor::new_tensor_handle(std::move(x)));
        stack[0] = from(raii.release());
        aoti_torch_call_dispatcher("aten::floor", "", stack);
        RAIIATH res(to<AtenTensorHandle>(stack[0]));
        return *reinterpret_cast<at::Tensor*>(res.release());
        }
        """
        module = torch.utils.cpp_extension.load_inline(
            name="inline_extension_using_shim_dispatcher",
            cpp_sources=[source],
            functions=["my_abs", "my_floor"],
        )

        t = torch.rand(2, 3) - 1.0
        floor_t = module.my_floor(t)
        abs_t = module.my_abs(t)
        self.assertEqual(abs_t, torch.abs(t))
        self.assertEqual(floor_t, torch.floor(t))

    @unittest.skipIf(not (TEST_CUDA or TEST_ROCM), "CUDA not found")
    def test_cuda_pluggable_allocator_include(self):
        """
        This method creates a minimal example to replicate the apex setup.py to build nccl_allocator extension
        """

        # the cpp source includes CUDAPluggableAllocator and has an empty exported function
        cpp_source = """
                #include <torch/csrc/cuda/CUDAPluggableAllocator.h>
                #include <torch/extension.h>
                int get_nccl_allocator() {
                    return 0;
                }
                PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                    m.def("get_nccl_allocator", []() { return get_nccl_allocator(); });
                }
                """

        build_dir = tempfile.mkdtemp()
        src_path = os.path.join(build_dir, "NCCLAllocator.cpp")

        with open(src_path, mode="w") as f:
            f.write(cpp_source)

        # initially success is false
        success = False
        try:
            # try to build the module
            torch.utils.cpp_extension.load(
                name="nccl_allocator",
                sources=src_path,
                verbose=True,
                with_cuda=True,
            )
            # set success as true if built successfully
            success = True
        except Exception as e:
            print(f"Failed to load the module: {e}")

        # test if build was successful
        self.assertEqual(success, True)


if __name__ == "__main__":
    common.run_tests()
