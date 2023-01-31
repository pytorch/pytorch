import os
import subprocess
import sys
import shutil
import contextlib


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


def append_multiple_lines(file_name, lines_to_append):
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        appendEOL = False
        # Move read cursor to the start of file.
        file_object.seek(0)
        # Check if file is not empty
        data = file_object.read(100)
        if len(data) > 0:
            appendEOL = True
        # Iterate over each string in the list
        for line in lines_to_append:
            # If file is not empty then append '\n' before first line for
            # other lines always append '\n' before appending line
            if appendEOL == True:
                file_object.write("\n")
            else:
                appendEOL = True
            # Append element at the end of file
            file_object.write(line)



if os.environ['DEBUG'] == '1':
    os.environ['BUILD_TYPE'] = 'debug'
else:
    os.environ['BUILD_TYPE'] = 'release'

if 'BUILD_ENVIRONMENT' not in os.environ:
    os.environ['CONDA_PARENT_DIR'] = str(os.getcwd())
else:
    os.environ['CONDA_PARENT_DIR'] = 'C:\\Jenkins'


os.environ['PATH'] = 'C:\\Program Files\\CMake\\bin;C:\\Program Files\\7-Zip;' +\
    'C:\\ProgramData\\chocolatey\\bin;C:\\Program Files\\Git\\cmd;C:\\Program Files' +\
        '\\Amazon\\AWSCLI;C:\\Program Files\\Amazon\\AWSCLI\\bin;' + os.environ['PATH']


'''
:: This inflates our log size slightly, but it is REALLY useful to be
:: able to see what our cl.exe commands are (since you can actually
:: just copy-paste them into a local Windows setup to just rebuild a
:: single file.)
:: log sizes are too long, but leaving this here incase someone wants to use it locally
:: set CMAKE_VERBOSE_MAKEFILE=1
'''


os.environ['INSTALLER_DIR'] = os.environ['SCRIPT_HELPERS_DIR'] + '\\installation-helpers'


subprocess.run('python ' + os.environ['INSTALLER_DIR'] + '\\install_mkl.py', shell=True)
subprocess.run('python ' + os.environ['INSTALLER_DIR'] + '\\install_magma.py', shell=True)
subprocess.run('python ' + os.environ['INSTALLER_DIR'] + '\\install_sccache.py', shell=True)

# test vars
os.environ['CMAKE_INCLUDE_PATH'] = os.environ['TMP_DIR_WIN'] + '\\mkl\\include'

if 'LIB' in os.environ:
    os.environ['LIB'] = os.environ['TMP_DIR_WIN'] + '\\mkl\\lib;' + os.environ['LIB']
else:
    os.environ['LIB'] = os.environ['TMP_DIR_WIN'] + '\\mkl\\lib'

if 'BUILD_ENVIRONMENT' not in os.environ:
    os.environ['CONDA_PARENT_DIR'] = str(os.getcwd())
else:
    os.environ['CONDA_PARENT_DIR'] = 'C:\\Jenkins'

os.environ['INSTALL_FRESH_CONDA'] = '1'

os.environ['PATH'] = os.environ['CONDA_PARENT_DIR'] + '\\Miniconda3\\Library\\bin;' + os.environ['CONDA_PARENT_DIR'] +\
    '\\Miniconda3;' + os.environ['CONDA_PARENT_DIR'] + '\\Miniconda3\\Scripts;' + os.environ['PATH']


'''
:: Miniconda has been installed as part of the Windows AMI with all the dependencies.
:: We just need to activate it here
'''
subprocess.run('python ' + os.environ['INSTALLER_DIR'] + '\\activate_miniconda3.py', shell=True)

try:
    os.environ['PATH'] = os.environ['CONDA_PARENT_DIR'] + '\\Miniconda3\\Library\\bin;' + os.environ['CONDA_PARENT_DIR'] +\
        '\\Miniconda3;' + os.environ['CONDA_PARENT_DIR'] + '\\Miniconda3\\Scripts;' + os.environ['PATH']

    result = subprocess.run('conda env list', shell=True)
    result.check_returncode()

except Exception as e:
    None

# Install ninja and other deps
if 'REBUILD' not in os.environ:
    subprocess.run('conda run -n test_env pip install -q ninja==1.10.0.post1 dataclasses typing_extensions expecttest==0.1.3', shell=True)

# Override VS env here
with pushd('.'):
    if 'VC_VERSION' not in os.environ:
        subprocess.run('\"C:\\Program Files (x86)\\Microsoft Visual Studio\\' +
            os.environ['VC_YEAR'] + '\\' + os.environ['VC_PRODUCT'] + '\\' +
                'VC\\Auxiliary\\Build\\vcvarsall.bat\" x64', shell=True)

    else:
        subprocess.run('\"C:\\Program Files (x86)\\Microsoft Visual Studio\\' +
            os.environ['VC_YEAR'] + '\\' + os.environ['VC_PRODUCT'] + '\\' +
                'VC\\Auxiliary\\Build\\vcvarsall.bat\" x64 -vcvars_ver=' + os.environ['VC_VERSION'], shell=True)


if os.environ['USE_CUDA'] == '1':

    os.environ['USE_CUDA'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v' + os.environ['CUDA_VERSION']

    # version transformer, for example 10.1 to 10_1.
    if '.' not in os.environ['CUDA_VERSION']:
        subprocess.run('echo CUDA version ' + cuda_version +
            'format isn\'t correct, which doesn\'t contain \'.\'', shell=True)

        sys.exit(1)

    # version transformer, for example 10.1 to 10_1.
    os.environ['VERSION_SUFFIX'] = str(os.environ['CUDA_VERSION']).replace('.', '_')
    os.environ['CUDA_PATH_V' + os.environ['VERSION_SUFFIX']] = os.environ['CUDA_PATH']

    os.environ['CUDNN_LIB_DIR'] = os.environ['CUDA_PATH'] + '\\lib\\x64'
    os.environ['CUDA_TOOLKIT_ROOT_DIR'] = os.environ['CUDA_PATH']
    os.environ['CUDNN_ROOT_DIR'] = os.environ['CUDA_PATH']
    os.environ['NVTOOLSEXT_PATH'] = 'C:\\Program Files\\NVIDIA Corporation\\NvToolsExt'
    os.environ['PATH'] = os.environ['CUDA_PATH'] + '\\bin;' + os.environ['CUDA_PATH'] +\
        '\\libnvvp;' + str(os.environ['PATH'])

    os.environ['CUDNN_LIB_DIR'] = os.environ['CUDA_PATH'] + '\\lib\\x64'
    os.environ['CUDA_TOOLKIT_ROOT_DIR'] = os.environ['CUDA_PATH']
    os.environ['CUDNN_ROOT_DIR'] = os.environ['CUDA_PATH']
    os.environ['NVTOOLSEXT_PATH'] = 'C:\\Program Files\\NVIDIA Corporation\\NvToolsExt'
    os.environ['PATH'] = os.environ['CUDA_PATH'] + '\\bin;' + os.environ['CUDA_PATH'] +\
        '\\libnvvp;' + os.environ['PATH']


os.environ['DISTUTILS_USE_SDK'] = '1'
os.environ['PATH'] = os.environ['TMP_DIR_WIN'] + '\\bin;' + os.environ['PATH']


'''
:: Target only our CI GPU machine's CUDA arch to speed up the build, we can overwrite with env var
:: default on circleci is Tesla T4 which has capability of 7.5, ref: https://developer.nvidia.com/cuda-gpus
:: jenkins has M40, which is 5.2
'''
if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
    os.environ['TORCH_CUDA_ARCH_LIST'] = '5.2'


# The default sccache idle timeout is 600, which is too short and leads to intermittent build errors.
os.environ['SCCACHE_IDLE_TIMEOUT'] = '0'
os.environ['SCCACHE_IGNORE_SERVER_IO_ERROR'] = '1'
subprocess.run('sccache --stop-server', shell=True)
subprocess.run('sccache --start-server', shell=True)
subprocess.run('sccache --zero-stats', shell=True)
os.environ['CC'] = 'sccache-cl'
os.environ['CXX'] = 'sccache-cl'

os.environ['CMAKE_GENERATOR'] = 'Ninja'


if os.environ['USE_CUDA'] == '1':
    '''
    :: randomtemp is used to resolve the intermittent build error related to CUDA.
    :: code: https://github.com/peterjc123/randomtemp-rust
    :: issue: https://github.com/pytorch/pytorch/issues/25393
    ::
    :: CMake requires a single command as CUDA_NVCC_EXECUTABLE, so we push the wrappers
    :: randomtemp.exe and sccache.exe into a batch file which CMake invokes.
    '''

    subprocess.run('curl -kL https://github.com/peterjc123/randomtemp-rust/releases/download/v0.4/randomtemp.exe --output ' +
        os.environ['TMP_DIR_WIN'] + '\\bin\\randomtemp.exe', shell=True)

    subprocess.run('echo @\"' + os.environ['TMP_DIR_WIN'] + '\\bin\\randomtemp.exe\" \"' +
        os.environ['TMP_DIR_WIN'] + '\\bin\\sccache.exe\" \"' + os.environ['CUDA_PATH'] +
            '\\bin\\nvcc.exe\" %%* > \"' + os.environ['TMP_DIR'] + '/bin/nvcc.bat\"', shell=True)

    subprocess.run('cat ' + os.environ['TMP_DIR'] + '/bin/nvcc.bat', shell=True)

    os.environ['CUDA_NVCC_EXECUTABLE'] = os.environ['TMP_DIR'] + '/bin/nvcc.bat'
    os.environ['CMAKE_CUDA_COMPILER'] = (os.environ['CUDA_PATH'] + '\\bin\\nvcc.exe').replace('\\', '/')
    os.environ['CMAKE_CUDA_COMPILER_LAUNCHER'] = os.environ['TMP_DIR']+ '/bin/randomtemp.exe;' +\
        os.environ['TMP_DIR'] + '\\bin\\sccache.exe'


subprocess.run('echo @echo off >> ' + os.environ['TMP_DIR_WIN'] +
    '\\ci_scripts\\pytorch_env_restore.bat', shell=True)

env_arr = []

for k, v in os.environ.items():
    env_arr.append(f"set {k}={v}")

append_multiple_lines(os.environ['TMP_DIR_WIN'] + '\\ci_scripts\\pytorch_env_restore.bat', env_arr)


if 'REBUILD' not in os.environ and 'BUILD_ENVIRONMENT' in os.environ:

    # Create a shortcut to restore pytorch environment
    subprocess.run('echo @echo off >> ' + os.environ['TMP_DIR_WIN'] +
        '/ci_scripts/pytorch_env_restore_helper.bat', shell=True)

    subprocess.run('echo call \"' + os.environ['TMP_DIR_WIN'] + '/ci_scripts/pytorch_env_restore.bat\" >> ' +
        os.environ['TMP_DIR_WIN'] + '/ci_scripts/pytorch_env_restore_helper.bat', shell=True)

    subprocess.run('echo cd /D \"%CD%\" >> ' + os.environ['TMP_DIR_WIN'] + '/ci_scripts/pytorch_env_restore_helper.bat', shell=True)

    subprocess.run('aws s3 cp \"s3://ossci-windows/Restore PyTorch Environment.lnk\" \"C:\\Users\\circleci\\Desktop\\Restore PyTorch Environment.lnk\"', shell=True)

subprocess.run('echo ' + str(os.environ), shell=True)
subprocess.run('env', shell=True)

subprocess.run('conda run -n test_env' + " python setup.py bdist_wheel", shell=True)

subprocess.run("sccache --show-stats", shell=True)
subprocess.run('conda run -n test_env' + ' python -c \"import os, glob; os.system(\'python -mpip install \'' +
    ' + glob.glob(\'dist/*.whl\')[0] + \'[opt-einsum]\')\"', shell=True)


if 'BUILD_ENVIRONMENT' not in os.environ:
    subprocess.run('echo NOTE: To run \'import torch\', please make sure to activate the conda environment by running \'call ' +
        os.environ['CONDA_PARENT_DIR'] + '\\Miniconda3\\Scripts\\activate.bat ' + os.environ['CONDA_PARENT_DIR'] + '\\Miniconda3\'' +
            ' in Command Prompt before running Git Bash.', shell=True)

else:
    subprocess.run('7z a ' + os.environ['TMP_DIR_WIN'] + '\\' + os.environ['IMAGE_COMMIT_TAG'] + '.7z ' +
        os.environ['CONDA_PARENT_DIR'] + '\\Miniconda3\\Lib\\site-packages\\torch ' + os.environ['CONDA_PARENT_DIR'] +
            '\\Miniconda3\\Lib\\site-packages\\torchgen ' + os.environ['CONDA_PARENT_DIR'] +
                '\\Miniconda3\\Lib\\site-packages\\functorch && copy /Y \"' + os.environ['TMP_DIR_WIN'] +
                    '\\' + os.environ['IMAGE_COMMIT_TAG'] + '.7z\" \"' + os.environ['PYTORCH_FINAL_PACKAGE_DIR'] + '\\\"', shell=True)

    # export test times so that potential sharded tests that'll branch off this build will use consistent data
    subprocess.run('conda run -n test_env python tools/stats/export_test_times.py', shell=True)
    shutil.copy(".pytorch-test-times.json", os.environ['PYTORCH_FINAL_PACKAGE_DIR'])

    # Also save build/.ninja_log as an artifact
    shutil.copy("build\\.ninja_log", os.environ['PYTORCH_FINAL_PACKAGE_DIR'] + '\\')



subprocess.run('sccache --show-stats --stats-format json | jq .stats > sccache-stats-' +
    os.environ['BUILD_ENVIRONMENT'] + '-' + os.environ['OUR_GITHUB_JOB_ID'] + '.json', shell=True)

subprocess.run('sccache --stop-server', shell=True)
