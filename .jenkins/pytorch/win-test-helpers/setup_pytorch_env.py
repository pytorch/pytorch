import os
from os.path import exists
import subprocess
import sys
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


tmp_dir = os.environ['TMP_DIR']


if exists(tmp_dir + '/ci_scripts/pytorch_env_restore.bat'):

    subprocess.run(tmp_dir + '/ci_scripts/pytorch_env_restore.bat', shell=True)
    sys.exit(0)


os.environ['PATH'] = 'C:\\Program Files\\CMake\\bin;C:\\Program Files\\7-Zip;C:\\' +\
    'ProgramData\\chocolatey\\bin;C:\\Program Files\\Git\\cmd;C:\\Program Files\\Amazon\\' +\
        'AWSCLI;C:\\Program Files\\Amazon\\AWSCLI\\bin;' + os.environ['PATH']

# Install Miniconda3
os.environ['INSTALLER_DIR'] = os.environ['SCRIPT_HELPERS_DIR'] + '\\installation-helpers'

# Miniconda has been installed as part of the Windows AMI with all the dependencies.
# We just need to activate it here
try:
    result = subprocess.run('python ' + os.environ['INSTALLER_DIR'] + '\\activate_miniconda3.py', shell=True)
    result.check_returncode()

except Exception as e:

    subprocess.run('echo activate conda failed', shell=True)
    subprocess.run('echo ' + str(e), shell=True)
    sys.exit()

# extra conda dependencies for testing purposes
if 'BUILD_ENVIRONMENT' in os.environ:

    try:
        result = subprocess.run('conda install -n test_env -y -q mkl protobuf numba scipy=1.6.2 ' +
            'typing_extensions dataclasses', shell=True)
        result.check_returncode()
    except Exception as e:

        subprocess.run('echo conda install failed', shell=True)
        subprocess.run('echo ' + str(e), shell=True)
        sys.exit()


with pushd('.'):

    try:
        if 'VC_VERSION' not in os.environ:
            result = subprocess.run('C:\\Program Files (x86)\\Microsoft Visual Studio\\' +
                os.environ['VC_YEAT'] + '\\' + os.environ['VC_VERSION'] +
                    '\\VC\\Auxiliary\\Build\\vcvarsall.bat x64', shell=True)
            result.check_returncode()

        else:
            result = subprocess.run('C:\\Program Files (x86)\\Microsoft Visual Studio\\' +
                os.environ['VC_YEAT'] + '\\' + os.environ['VC_VERSION'] +
                    '\\VC\\Auxiliary\\Build\\vcvarsall.bat x64 -vcvars_ver=' + os.environ['VC_VERSION'], shell=True)
            result.check_returncode()


    except Exception as e:

        subprocess.run('echo vcvarsall failed', shell=True)
        subprocess.run('echo ' + str(e), shell=True)
        sys.exit()


# The version is fixed to avoid flakiness: https://github.com/pytorch/pytorch/issues/31136
# =======
# Pin unittest-xml-reporting to freeze printing test summary logic, related: https://github.com/pytorch/pytorch/issues/69014

try:
    result = subprocess.run('conda install -n test_env pip install ninja==1.10.0.post1 future ' +
        'hypothesis==5.35.1 expecttest==0.1.3 librosa>=0.6.2 scipy==1.6.3 psutil pillow ' +
            'unittest-xml-reporting<=3.2.0,>=2.0.0 pytest pytest-xdist pytest-shard pytest-rerunfailures ' +
                'sympy xdoctest==1.0.2 pygments==2.12.0 opt-einsum>=3.3', shell=True)
    result.check_returncode()

except Exception as e:

    subprocess.run('echo install dependencies failed', shell=True)
    subprocess.run('echo ' + str(e), shell=True)
    sys.exit()


os.environ['DISTUTILS_USE_SDK'] = '1'

if os.environ['USE_CUDA'] == '1':

    os.environ['CUDA_PATH'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v'\
        + os.environ['CUDA_VERSION']

    # version transformer, for example 10.1 to 10_1.
    os.environ['VERSION_SUFFIX'] = str(os.environ['CUDA_VERSION']).replace('.', '_')
    os.environ['CUDA_PATH_V' + str(os.environ['VERSION_SUFFIX'])] = str(os.environ['CUDA_PATH'])

    os.environ['CUDNN_LIB_DIR'] = str(os.environ['CUDA_PATH']) + '\\lib\\x64'
    os.environ['CUDA_TOOLKIT_ROOT_DIR'] = str(os.environ['CUDA_PATH'])
    os.environ['CUDNN_ROOT_DIR'] = str(os.environ['CUDA_PATH'])
    os.environ['NVTOOLSEXT_PATH'] = 'C:\\Program Files\\NVIDIA Corporation\\NvToolsExt'
    os.environ['PATH'] = str(os.environ['CUDA_PATH']) + '\\bin;' + str(os.environ['CUDA_PATH']) +\
        '\\libnvvp;' + str(os.environ['PATH'])
    os.environ['NUMBAPRO_CUDALIB'] = str(os.environ['CUDA_PATH']) + '\\bin'
    os.environ['NUMBAPRO_LIBDEVICE'] = str(os.environ['CUDA_PATH']) + '\\nvvm\\libdevice'
    os.environ['NUMBAPRO_NVVM'] = str(os.environ['CUDA_PATH']) + '\\nvvm\\bin\\nvvm64_32_0.dll'


os.environ['PYTHONPATH'] = str(os.environ['TMP_DIR_WIN']) + '\\build;' + str(os.environ['PYTHONPATH'])

if 'BUILD_ENVIRONMENT' in os.environ:

    with pushd(str(os.environ['TMP_DIR_WIN']) + '\\build'):

        subprocess.run('copy /Y ' + str(os.environ['PYTORCH_FINAL_PACKAGE_DIR_WIN']) +
            '\\' + str(os.environ['IMAGE_COMMIT_TAG']) + '.7z ' + str(os.environ['TMP_DIR_WIN']) + '\\', shell=True)

        # 7z: -aos skips if exists because this .bat can be called multiple times

        subprocess.run('7z x ' + str(os.environ['TMP_DIR_WIN']) + '\\' +
            str(os.environ['IMAGE_COMMIT_TAG']) + '.7z -aos', shell=True)

else:

    subprocess.run('xcopy /s ' + str(os.environ['CONDA_PARENT_DIR']) +
        '\\Miniconda3\\Lib\\site-packages\\torch ' + str(os.environ['TMP_DIR_WIN']) +
            '\\build\\torch\\', shell=True)

subprocess.run('echo @echo off >> ' + str(os.environ['TMP_DIR_WIN']) +
    '/ci_scripts/pytorch_env_restore.bat', shell=True)

env_arr = []

for k, v in os.environ.items():
    env_arr.append('set ' + k)

append_multiple_lines(os.environ['TMP_DIR_WIN'] + '/ci_scripts/pytorch_env_restore.bat', env_arr)


if 'BUILD_ENVIRONMENT' in os.environ:

    # Create a shortcut to restore pytorch environment
    subprocess.run('echo @echo off >> ' + str(os.environ['TMP_DIR_WIN']) +
        '/ci_scripts/pytorch_env_restore_helper.bat', shell=True)
    subprocess.run('echo call \"%TMP_DIR_WIN%/ci_scripts/pytorch_env_restore.bat\" >> ' +
        str(os.environ['TMP_DIR_WIN']) + '/ci_scripts/pytorch_env_restore_helper.bat', shell=True)
    subprocess.run('echo cd /D \"%CD%\" >> ' + str(os.environ['TMP_DIR_WIN']) +
        '/ci_scripts/pytorch_env_restore_helper.bat', shell=True)

    subprocess.run('aws s3 cp \"s3://ossci-windows/Restore PyTorch Environment.lnk\" ' +
        '\"C:\\Users\\circleci\\Desktop\\Restore PyTorch Environment.lnk\"', shell=True)
