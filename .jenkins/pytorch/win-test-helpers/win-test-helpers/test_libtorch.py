import os
import subprocess
import sys


def libtorch_check(file_name, file_full_path):

    # Skip verify_api_visibility as it a compile level test
    sys.exit(0) if file_name == "verify_api_visibility" else None

    # See https://github.com/pytorch/pytorch/issues/25161
    sys.exit(0) if file_name == "c10_metaprogramming_test" or file_name == "module_test" else None

    # See https://github.com/pytorch/pytorch/issues/25312
    sys.exit(0) if file_name == "converter_nomigraph_test" else None

    # See https://github.com/pytorch/pytorch/issues/35636
    sys.exit(0) if file_name == "generate_proposals_op_gpu_test" else None

    # See https://github.com/pytorch/pytorch/issues/35648
    sys.exit(0) if file_name == "reshape_op_gpu_test" else None

    # See https://github.com/pytorch/pytorch/issues/35651
    sys.exit(0) if file_name == "utility_ops_gpu_test" else None

    subprocess.run('echo Running ' + file_full_path, shell=True)
    if file_name == "c10_intrusive_ptr_benchmark":
        subprocess.run(file_full_path, shell=True)
        sys.exit(0)

    # Differentiating the test report directories is crucial for test time reporting.
    os.mkdir(os.environ['TEST_OUT_DIR'] + '\\' + file_name + '.exe')
    subprocess.run(file_full_path + ' --gtest_output=xml:' + os.environ['TEST_OUT_DIR'] +
        '\\' + file_name + '.exe' + '\\' + file_name + '.xml', shell=True)


# Skip LibTorch tests when building a GPU binary and testing on a CPU machine
# because LibTorch tests are not well designed for this use case.


if os.environ['USE_CUDA'] == '0' and not os.environ['CUDA_VERSION'] == 'cpu':
    sys.exit(0)

subprocess.run('python ' + os.environ['SCRIPT_HELPERS_DIR'] + '\\setup_pytorch_env.py', shell=True)

os.chdir(os.environ['TMP_DIR_WIN'] + '\\build\\torch\\bin')
os.environ['TMP_DIR_WIN'] = 'dp0\\..\\..\\..\\test\\test-reports\\cpp-unittest'
os.mkdir("dp0\\..\\..\\..\\test\\test-reports\\cpp-unittest")
os.environ['PATH'] = 'C:\\Program Files\\NVIDIA Corporation\\NvToolsExt\\bin\\x64;' +\
    os.environ['TMP_DIR_WIN'] + '\\build\\torch\\lib;' + os.environ['PATH']

os.environ['TEST_API_OUT_DIR'] = os.environ['TEST_OUT_DIR'] + '\\test_api'
os.mkdir(os.environ['TEST_API_OUT_DIR'])
subprocess.run('test_api.exe --gtest_filter="-IntegrationTest.MNIST*" --gtest_output=xml:' +
    os.environ['TEST_API_OUT_DIR'] + '\\test_api.xml', shell=True)

os.chdir(os.environ['TMP_DIR_WIN'] + '\\build\\torch\\test')

for file in os.listdir('.'):
    if file.endswith('.exe'):
        libtorch_check(file.split('.')[0], os.path.abspath(file))
