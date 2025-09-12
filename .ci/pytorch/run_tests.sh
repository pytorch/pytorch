#!/bin/bash
# shellcheck disable=SC2086,SC2048,SC2068,SC2145,SC2034,SC2207,SC2143
# TODO: Re-enable shellchecks above

set -eux -o pipefail

# Essentially runs pytorch/test/run_test.py, but keeps track of which tests to
# skip in a centralized place.
#
# TODO Except for a few tests, this entire file is a giant TODO. Why are these
# tests # failing?
# TODO deal with Windows

# This script expects to be in the pytorch root folder
if [[ ! -d 'test' || ! -f 'test/run_test.py' ]]; then
    echo "run_tests.sh expects to be run from the Pytorch root directory " \
         "but I'm actually in $(pwd)"
    exit 2
fi

# Allow master skip of all tests
if [[ -n "${SKIP_ALL_TESTS:-}" ]]; then
    exit 0
fi

# If given specific test params then just run those
if [[ -n "${RUN_TEST_PARAMS:-}" ]]; then
    echo "$(date) :: Calling user-command $(pwd)/test/run_test.py ${RUN_TEST_PARAMS[@]}"
    python test/run_test.py ${RUN_TEST_PARAMS[@]}
    exit 0
fi

# Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

# Parameters
##############################################################################
if [[ "$#" != 3 ]]; then
  if [[ -z "${DESIRED_PYTHON:-}" || -z "${DESIRED_CUDA:-}" || -z "${PACKAGE_TYPE:-}" ]]; then
    echo "USAGE: run_tests.sh  PACKAGE_TYPE  DESIRED_PYTHON  DESIRED_CUDA"
    echo "The env variable PACKAGE_TYPE must be set to 'manywheel' or 'libtorch'"
    echo "The env variable DESIRED_PYTHON must be set like '2.7mu' or '3.6m' etc"
    echo "The env variable DESIRED_CUDA must be set like 'cpu' or 'cu80' etc"
    exit 1
  fi
  package_type="$PACKAGE_TYPE"
  py_ver="$DESIRED_PYTHON"
  cuda_ver="$DESIRED_CUDA"
else
  package_type="$1"
  py_ver="$2"
  cuda_ver="$3"
fi

if [[ "$cuda_ver" == 'cpu-cxx11-abi' ]]; then
    cuda_ver="cpu"
fi

# cu80, cu90, cu100, cpu
if [[ ${#cuda_ver} -eq 4 ]]; then
    cuda_ver_majmin="${cuda_ver:2:1}.${cuda_ver:3:1}"
elif [[ ${#cuda_ver} -eq 5 ]]; then
    cuda_ver_majmin="${cuda_ver:2:2}.${cuda_ver:4:1}"
fi

NUMPY_PACKAGE=""
if [[ ${py_ver} == "3.10" ]]; then
    PROTOBUF_PACKAGE="protobuf>=3.17.2"
    NUMPY_PACKAGE="numpy>=1.21.2"
else
    PROTOBUF_PACKAGE="protobuf=3.14.0"
fi

# Environment initialization
retry pip install -qUr requirements-build.txt
# AArch64 requires newer numpy version
if [[ "$(uname)" == Darwin || "$(uname -m)" == aarch64 ]]; then
    # Install the testing dependencies
    retry pip install -q future hypothesis ${NUMPY_PACKAGE} ${PROTOBUF_PACKAGE} pytest
else
    retry pip install -qr requirements.txt || true
    retry pip install -q hypothesis protobuf pytest || true
    numpy_ver=1.15
    case "$(python --version 2>&1)" in
      *2* | *3.5* | *3.6*)
        numpy_ver=1.11
        ;;
    esac
    retry pip install -q "numpy==${numpy_ver}" || true
fi

echo "Testing with:"
pip freeze

##############################################################################
# Smoke tests
##############################################################################
# TODO use check_binary.sh, which requires making sure it runs on Windows
pushd /
echo "Smoke testing imports"
python -c 'import torch'

# Check for MKL and MKLDNN (AArch64)
if [[ "$(uname)" != 'Darwin' || "$PACKAGE_TYPE" != *wheel ]]; then
    if [[ "$(uname -m)" == "aarch64" ]]; then
        echo "Checking that MKLDNN is available on AArch64"
        python -c 'import torch; exit(0 if torch.backends.mkldnn.is_available() else 1)'
    else
        echo "Checking that MKL is available"
        python -c 'import torch; exit(0 if torch.backends.mkl.is_available() else 1)'
    fi
else
    echo 'Not checking for MKL on Darwin wheel packages'
fi

if [[ "$OSTYPE" == "msys" ]]; then
    GPUS=$(wmic path win32_VideoController get name)
    if [[ ! "$GPUS" == *NVIDIA* ]]; then
        echo "Skip CUDA tests for machines without a Nvidia GPU card"
        exit 0
    fi
fi

# Test that the version number is consistent during building and testing
if [[ "$PYTORCH_BUILD_NUMBER" -gt 1 ]]; then
    expected_version="${PYTORCH_BUILD_VERSION}.post${PYTORCH_BUILD_NUMBER}"
else
    expected_version="${PYTORCH_BUILD_VERSION}"
fi
echo "Checking that we are testing the package that is just built"
python -c "import torch; exit(0 if torch.__version__ == '$expected_version' else 1)"

# Test that CUDA builds are setup correctly
if [[ "$cuda_ver" != 'cpu' ]]; then
    cuda_installed=1
    nvidia-smi || cuda_installed=0
    if [[ "$cuda_installed" == 0 ]]; then
      echo "Skip CUDA tests for machines without a Nvidia GPU card"
    else
      # Test CUDA archs
      echo "Checking that CUDA archs are setup correctly"
      timeout 20 python -c 'import torch; torch.randn([3,5]).cuda()'

      # These have to run after CUDA is initialized
      echo "Checking that magma is available"
      python -c 'import torch; torch.rand(1).cuda(); exit(0 if torch.cuda.has_magma else 1)'
      echo "Checking that CuDNN is available"
      python -c 'import torch; exit(0 if torch.backends.cudnn.is_available() else 1)'
    fi
fi

# Check that OpenBlas is not linked to on MacOS
if [[ "$(uname)" == 'Darwin' ]]; then
    echo "Checking the OpenBLAS is not linked to"
    all_dylibs=($(find "$(python -c "import site; print(site.getsitepackages()[0])")"/torch -name '*.dylib'))
    for dylib in "${all_dylibs[@]}"; do
        if [[ -n "$(otool -L $dylib | grep -i openblas)" ]]; then
            echo "Found openblas as a dependency of $dylib"
            echo "Full dependencies is: $(otool -L $dylib)"
            exit 1
        fi
    done

    echo "Checking that OpenMP is available"
    python -c "import torch; exit(0 if torch.backends.openmp.is_available() else 1)"
fi

popd

# TODO re-enable the other tests after the nightlies are moved to CI. This is
# because the binaries keep breaking, often from additional tests, that aren't
# real problems. Once these are on circleci and a smoke-binary-build is added
# to PRs then this should stop happening and these can be re-enabled.
echo "Not running unit tests. Hopefully these problems are caught by CI"
exit 0


##############################################################################
# Running unit tests (except not right now)
##############################################################################
echo "$(date) :: Starting tests for $package_type package for python$py_ver and $cuda_ver"

# We keep track of exact tests to skip, as otherwise we would be hardly running
# any tests. But b/c of issues working with pytest/normal-python-test/ and b/c
# of special snowflake tests in test/run_test.py we also take special care of
# those
tests_to_skip=()

#
# Entire file exclusions
##############################################################################
entire_file_exclusions=("-x")

# cpp_extensions doesn't work with pytest, so we exclude it from the pytest run
# here and then manually run it later. Note that this is only because this
# entire_fil_exclusions flag is only passed to the pytest run
entire_file_exclusions+=("cpp_extensions")

# TODO temporary line to fix next days nightlies, but should be removed when
# issue is fixed
entire_file_exclusions+=('type_info')

if [[ "$cuda_ver" == 'cpu' ]]; then
    # test/test_cuda.py exits early if the installed torch is not built with
    # CUDA, but the exit doesn't work when running with pytest, so pytest will
    # still try to run all the CUDA tests and then fail
    entire_file_exclusions+=("cuda")
    entire_file_exclusions+=("nccl")
fi

if [[ "$(uname)" == 'Darwin' || "$OSTYPE" == "msys" ]]; then
    # pytest on Mac doesn't like the exits in these files
    entire_file_exclusions+=('c10d')
    entire_file_exclusions+=('distributed')

    # pytest doesn't mind the exit but fails the tests. On Mac we run this
    # later without pytest
    entire_file_exclusions+=('thd_distributed')
fi


#
# Universal flaky tests
##############################################################################

# RendezvousEnvTest sometimes hangs forever
# Otherwise it will fail on CUDA with
#   Traceback (most recent call last):
#     File "test_c10d.py", line 179, in test_common_errors
#       next(gen)
#   AssertionError: ValueError not raised
tests_to_skip+=('RendezvousEnvTest and test_common_errors')

# This hung forever once on conda_3.5_cu92
tests_to_skip+=('TestTorch and test_sum_dim')

# test_trace_warn isn't actually flaky, but it doesn't work with pytest so we
# just skip it
tests_to_skip+=('TestJit and test_trace_warn')
#
# Python specific flaky tests
##############################################################################

# test_dataloader.py:721: AssertionError
# looks like a timeout, but interestingly only appears on python 3
if [[ "$py_ver" == 3* ]]; then
    tests_to_skip+=('TestDataLoader and test_proper_exit')
fi

#
# CUDA flaky tests, all package types
##############################################################################
if [[ "$cuda_ver" != 'cpu' ]]; then

    #
    # DistributedDataParallelTest
    # All of these seem to fail
    tests_to_skip+=('DistributedDataParallelTest')

    #
    # RendezvousEnvTest
    # Traceback (most recent call last):
    #   File "test_c10d.py", line 201, in test_nominal
    #     store0, rank0, size0 = next(gen0)
    #   File "/opt/python/cp36-cp36m/lib/python3.6/site-packages/torch/distributed/rendezvous.py", line 131, in _env_rendezvous_handler
    #     store = TCPStore(master_addr, master_port, start_daemon)
    # RuntimeError: Address already in use
    tests_to_skip+=('RendezvousEnvTest and test_nominal')

    #
    # TestCppExtension
    #
    # Traceback (most recent call last):
    #   File "test_cpp_extensions.py", line 134, in test_jit_cudnn_extension
    #     with_cuda=True)
    #   File "/opt/python/cp35-cp35m/lib/python3.5/site-packages/torch/utils/cpp_extension.py", line 552, in load
    #     with_cuda)
    #   File "/opt/python/cp35-cp35m/lib/python3.5/site-packages/torch/utils/cpp_extension.py", line 729, in _jit_compile
    #     return _import_module_from_library(name, build_directory)
    #   File "/opt/python/cp35-cp35m/lib/python3.5/site-packages/torch/utils/cpp_extension.py", line 867, in _import_module_from_library
    #     return imp.load_module(module_name, file, path, description)
    #   File "/opt/python/cp35-cp35m/lib/python3.5/imp.py", line 243, in load_module
    #     return load_dynamic(name, filename, file)
    #   File "/opt/python/cp35-cp35m/lib/python3.5/imp.py", line 343, in load_dynamic
    #     return _load(spec)
    #   File "<frozen importlib._bootstrap>", line 693, in _load
    #   File "<frozen importlib._bootstrap>", line 666, in _load_unlocked
    #   File "<frozen importlib._bootstrap>", line 577, in module_from_spec
    #   File "<frozen importlib._bootstrap_external>", line 938, in create_module
    #   File "<frozen importlib._bootstrap>", line 222, in _call_with_frames_removed
    # ImportError: libcudnn.so.7: cannot open shared object file: No such file or directory
    tests_to_skip+=('TestCppExtension and test_jit_cudnn_extension')

    #
    # TestCuda
    #

    # 3.7_cu80
    #  RuntimeError: CUDA error: out of memory
    tests_to_skip+=('TestCuda and test_arithmetic_large_tensor')

    # 3.7_cu80
    # RuntimeError: cuda runtime error (2) : out of memory at /opt/conda/conda-bld/pytorch-nightly_1538097262541/work/aten/src/THC/THCTensorCopy.cu:205
    tests_to_skip+=('TestCuda and test_autogpu')

    #
    # TestDistBackend
    #

    # Traceback (most recent call last):
    #   File "test_thd_distributed.py", line 1046, in wrapper
    #     self._join_and_reduce(fn)
    #   File "test_thd_distributed.py", line 1108, in _join_and_reduce
    #     self.assertEqual(p.exitcode, first_process.exitcode)
    #   File "/pytorch/test/common.py", line 399, in assertEqual
    #     super(TestCase, self).assertEqual(x, y, message)
    # AssertionError: None != 77 :
    tests_to_skip+=('TestDistBackend and test_all_gather_group')
    tests_to_skip+=('TestDistBackend and test_all_reduce_group_max')
    tests_to_skip+=('TestDistBackend and test_all_reduce_group_min')
    tests_to_skip+=('TestDistBackend and test_all_reduce_group_sum')
    tests_to_skip+=('TestDistBackend and test_all_reduce_group_product')
    tests_to_skip+=('TestDistBackend and test_barrier_group')
    tests_to_skip+=('TestDistBackend and test_broadcast_group')

    # Traceback (most recent call last):
    #   File "test_thd_distributed.py", line 1046, in wrapper
    #     self._join_and_reduce(fn)
    #   File "test_thd_distributed.py", line 1108, in _join_and_reduce
    #     self.assertEqual(p.exitcode, first_process.exitcode)
    #   File "/pytorch/test/common.py", line 397, in assertEqual
    #     super(TestCase, self).assertLessEqual(abs(x - y), prec, message)
    # AssertionError: 12 not less than or equal to 1e-05
    tests_to_skip+=('TestDistBackend and test_barrier')

    # Traceback (most recent call last):
    #   File "test_distributed.py", line 1267, in wrapper
    #     self._join_and_reduce(fn)
    #   File "test_distributed.py", line 1350, in _join_and_reduce
    #     self.assertEqual(p.exitcode, first_process.exitcode)
    #   File "/pytorch/test/common.py", line 399, in assertEqual
    #     super(TestCase, self).assertEqual(x, y, message)
    # AssertionError: None != 1
    tests_to_skip+=('TestDistBackend and test_broadcast')

    # Memory leak very similar to all the conda ones below, but appears on manywheel
    # 3.6m_cu80
    # AssertionError: 1605632 not less than or equal to 1e-05 : __main__.TestEndToEndHybridFrontendModels.test_vae_cuda leaked 1605632 bytes CUDA memory on device 0
    tests_to_skip+=('TestEndToEndHybridFrontendModels and test_vae_cuda')

    # ________________________ TestNN.test_embedding_bag_cuda ________________________
    #
    # self = <test_nn.TestNN testMethod=test_embedding_bag_cuda>
    # dtype = torch.float32
    #
    #     @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    #     @repeat_test_for_types(ALL_TENSORTYPES)
    #     @skipIfRocm
    #     def test_embedding_bag_cuda(self, dtype=torch.float):
    #         self._test_EmbeddingBag(True, 'sum', False, dtype)
    #         self._test_EmbeddingBag(True, 'mean', False, dtype)
    #         self._test_EmbeddingBag(True, 'max', False, dtype)
    #         if dtype != torch.half:
    #             # torch.cuda.sparse.HalfTensor is not enabled.
    #             self._test_EmbeddingBag(True, 'sum', True, dtype)
    # >           self._test_EmbeddingBag(True, 'mean', True, dtype)
    #
    # test_nn.py:2144:
    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    # test_nn.py:2062: in _test_EmbeddingBag
    #     _test_vs_Embedding(N, D, B, L)
    # test_nn.py:2059: in _test_vs_Embedding
    #     self.assertEqual(es_weight_grad, e.weight.grad, needed_prec)
    # common.py:373: in assertEqual
    #     assertTensorsEqual(x, y)
    # common.py:365: in assertTensorsEqual
    #     self.assertLessEqual(max_err, prec, message)
    # E   AssertionError: tensor(0.0000, device='cuda:0', dtype=torch.float32) not less than or equal to 2e-05 :
    #  1 failed, 1202 passed, 19 skipped, 2 xfailed, 796 warnings in 1166.73 seconds =
    # Traceback (most recent call last):
    #   File "test/run_test.py", line 391, in <module>
    #     main()
    #   File "test/run_test.py", line 383, in main
    #     raise RuntimeError(message)
    tests_to_skip+=('TestNN and test_embedding_bag_cuda')
fi

##############################################################################
# MacOS specific flaky tests
##############################################################################

if [[ "$(uname)" == 'Darwin' ]]; then
    # TestCppExtensions by default uses a temp folder in /tmp. This doesn't
    # work for this Mac machine cause there is only one machine and /tmp is
    # shared. (All the linux builds are on docker so have their own /tmp).
    tests_to_skip+=('TestCppExtension')
fi

# Turn the set of tests to skip into an invocation that pytest understands
excluded_tests_logic=''
for exclusion in "${tests_to_skip[@]}"; do
    if [[ -z "$excluded_tests_logic" ]]; then
        # Only true for i==0
        excluded_tests_logic="not ($exclusion)"
    else
        excluded_tests_logic="$excluded_tests_logic and not ($exclusion)"
    fi
done


##############################################################################
# Run the tests
##############################################################################
echo
echo "$(date) :: Calling 'python test/run_test.py -v -p pytest ${entire_file_exclusions[@]} -- --disable-pytest-warnings -k '$excluded_tests_logic'"

python test/run_test.py -v -p pytest ${entire_file_exclusions[@]} -- --disable-pytest-warnings -k "'" "$excluded_tests_logic" "'"

echo
echo "$(date) :: Finished 'python test/run_test.py -v -p pytest ${entire_file_exclusions[@]} -- --disable-pytest-warnings -k '$excluded_tests_logic'"

# cpp_extensions don't work with pytest, so we run them without pytest here,
# except there's a failure on CUDA builds (documented above), and
# cpp_extensions doesn't work on a shared mac machine (also documented above)
if [[ "$cuda_ver" == 'cpu' && "$(uname)" != 'Darwin' ]]; then
    echo
    echo "$(date) :: Calling 'python test/run_test.py -v -i cpp_extensions'"
    python test/run_test.py -v -i cpp_extensions
    echo
    echo "$(date) :: Finished 'python test/run_test.py -v -i cpp_extensions'"
fi

# thd_distributed can run on Mac but not in pytest
if [[ "$(uname)" == 'Darwin' ]]; then
    echo
    echo "$(date) :: Calling 'python test/run_test.py -v -i thd_distributed'"
    python test/run_test.py -v -i thd_distributed
    echo
    echo "$(date) :: Finished 'python test/run_test.py -v -i thd_distributed'"
fi
