import os
from os.path import exists
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



if os.environ['DEBUG'] == '1':
    os.environ['BUILD_TYPE'] = 'debug'
else:
    os.environ['BUILD_TYPE'] = 'release'


os.environ['PATH'] = 'C:\\Program Files\\CMake\\bin;C:\\Program Files\\7-Zip;'+\
'C:\\ProgramData\\chocolatey\\bin;C:\\Program Files\\Git\\cmd;C:\\Program Files'+\
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


subprocess.run(['python', os.environ['INSTALLER_DIR'] + '\\install_mkl.py'])
subprocess.run(['python',os.environ['INSTALLER_DIR'] + '\\install_magma.py'])
subprocess.run(['python',os.environ['INSTALLER_DIR'] + '\\install_sccache.py'])

'''
:: Miniconda has been installed as part of the Windows AMI with all the dependencies.
:: We just need to activate it here
'''
subprocess.run(['python',os.environ['INSTALLER_DIR'] + '\\activate_miniconda3.py'])

# Install ninja and other deps
if 'REBUILD' not in os.environ:
    subprocess.run(['pip', 'install', '-q', "ninja==1.10.0.post1", 'dataclasses',\
     'typing_extensions', "expecttest==0.1.3"])

# Override VS env here
with pushd('.'):
    if 'VC_VERSION' not in os.environ:
        subprocess.call('C:\\Program Files (x86)\\Microsoft Visual Studio\\' +\
        os.environ['VC_YEAR'] + '\\' + os.environ['VC_PRODUCT'] + '\\' +\
        'VC\Auxiliary\Build\vcvarsall.bat x64', shell=True)

    else:
        subprocess.call('C:\\Program Files (x86)\\Microsoft Visual Studio\\' +\
        os.environ['VC_YEAR'] + '\\' + os.environ['VC_PRODUCT'] + '\\' +\
        'VC\Auxiliary\Build\vcvarsall.bat x64 -vcvars_ver=' + os.environ['VC_VERSION'], shell=True)

    subprocess.call('@echo on', shell=True)


if os.environ['USE_CUDA'] == '1':

    os.environ['USE_CUDA'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v' + os.environ['CUDA_VERSION']

    # version transformer, for example 10.1 to 10_1.
    if not os.environ['CUDA_VERSION'].contains('.'):
        subprocess.run(['echo', 'CUDA version ' + cuda_version +\
         'format isn\'t correct, which doesn\'t contain \'.\''])

        sys.exit(1)

    # version transformer, for example 10.1 to 10_1.
    os.environ['VERSION_SUFFIX']=str(os.environ['CUDA_VERSION']).replace('.','_')
    os.environ['CUDA_PATH_V' + str(os.environ['VERSION_SUFFIX'])]=str(os.environ['CUDA_PATH'])

    os.environ['CUDNN_LIB_DIR']=str(os.environ['CUDA_PATH']) + '\\lib\\x64'
    os.environ['CUDA_TOOLKIT_ROOT_DIR']=str(os.environ['CUDA_PATH'])
    os.environ['CUDNN_ROOT_DIR']=str(os.environ['CUDA_PATH'])
    os.environ['NVTOOLSEXT_PATH']='C:\\Program Files\\NVIDIA Corporation\\NvToolsExt'
    os.environ['PATH']=str(os.environ['CUDA_PATH']) + '\\bin;' + str(os.environ['CUDA_PATH'])+\
    '\\libnvvp;' + str(os.environ['PATH'])

    os.environ['CUDNN_LIB_DIR']=os.environ['CUDA_PATH'] + '\\lib\\x64'
    os.environ['CUDA_TOOLKIT_ROOT_DIR']=os.environ['CUDA_PATH']
    os.environ['CUDNN_ROOT_DIR']=os.environ['CUDA_PATH']
    os.environ['NVTOOLSEXT_PATH']='C:\\Program Files\\NVIDIA Corporation\\NvToolsExt'
    os.environ['PATH']=os.environ['CUDA_PATH'] + '\\bin;' + os.environ['CUDA_PATH'] +\
     '\\libnvvp;' + os.environ['PATH']


os.environ['DISTUTILS_USE_SDK']='1'
os.environ['PATH']=os.environ['TMP_DIR_WIN'] + '\\bin;' + os.environ['PATH']


'''
:: Target only our CI GPU machine's CUDA arch to speed up the build, we can overwrite with env var
:: default on circleci is Tesla T4 which has capability of 7.5, ref: https://developer.nvidia.com/cuda-gpus
:: jenkins has M40, which is 5.2
'''
os.environ['TORCH_CUDA_ARCH_LIST']='5.2' if os.environ['TORCH_CUDA_ARCH_LIST'] == "" else None


# The default sccache idle timeout is 600, which is too short and leads to intermittent build errors.
os.environ['SCCACHE_IDLE_TIMEOUT']='0'
os.environ['SCCACHE_IGNORE_SERVER_IO_ERROR']='1'
subprocess.run(['sccache', '--stop-server'])
subprocess.run(['sccache', '--start-server'])
subprocess.run(['sccache', '--zero-stats'])
os.environ['CC']='sccache-cl'
os.environ['CXX']='sccache-cl'

os.environ['CMAKE_GENERATOR']='Ninja'


if os.environ['USE_CUDA']=='1':
    '''
    :: randomtemp is used to resolve the intermittent build error related to CUDA.
    :: code: https://github.com/peterjc123/randomtemp-rust
    :: issue: https://github.com/pytorch/pytorch/issues/25393
    ::
    :: CMake requires a single command as CUDA_NVCC_EXECUTABLE, so we push the wrappers
    :: randomtemp.exe and sccache.exe into a batch file which CMake invokes.
    '''

    subprocess.run(['curl', '-kL', 'https://github.com/peterjc123/randomtemp-rust/releases/download/v0.4/randomtemp.exe',\
     '--output', os.environ['TMP_DIR_WIN'] + '\\bin\\randomtemp.exe'])

    subprocess.call('echo \@\"' + os.environ['TMP_DIR_WIN'] + '\\bin\\randomtemp.exe\" \"' +\
    os.environ['TMP_DIR_WIN'] + '\\bin\\sccache.exe\" \"' + os.environ['CUDA_PATH'] +\
    '\\bin\\nvcc.exe\" \%\%\* \> \"' + os.environ['TMP_DIR'] + '/bin/nvcc.bat\"', shell=True)

    subprocess.call('cat ' + os.environ['TMP_DIR'] + '/bin/nvcc.bat', shell=True)

    os.environ['CUDA_NVCC_EXECUTABLE']=os.environ['TMP_DIR'] + '/bin/nvcc.bat'
    os.environ['CMAKE_CUDA_COMPILER']=(os.environ['CUDA_PATH'] + '\\bin\\nvcc.exe').replace('\\', '/')
    os.environ['CMAKE_CUDA_COMPILER_LAUNCHER']=os.environ['TMP_DIR']+ '/bin/randomtemp.exe;' +\
    os.environ['TMP_DIR'] + '\\bin\\sccache.exe'


subprocess.run(['@echo', 'off'])
subprocess.run(['echo', '@echo', 'off', '>>', os.environ['TMP_DIR_WIN'] +\
'\\ci_scripts\\pytorch_env_restore.bat'])

restore_file = open(str(os.environ['TMP_DIR_WIN']) + '\\ci_scripts\\pytorch_env_restore.bat', 'a+')
set_file = open('set', 'r')
restore_file.write(set_file.read())
restore_file.close()
set_file.close()

subprocess.run(['@echo', 'on'])


if 'REBUILD' not in os.environ and 'BUILD_ENVIRONMENT' in os.environ:

    # Create a shortcut to restore pytorch environment
    subprocess.run(['echo', '@echo', 'off', '>>', os.environ['TMP_DIR_WIN'] +\
    '/ci_scripts/pytorch_env_restore_helper.bat'])

    subprocess.run(['echo', 'call', '\"' + os.environ['TMP_DIR_WIN'] + '/ci_scripts/pytorch_env_restore.bat\"',\
    '>>', os.environ['TMP_DIR_WIN'] + '/ci_scripts/pytorch_env_restore_helper.bat'])

    subprocess.run(['echo', 'cd', '/D', '\"' + os.environ['CD'] + '\"', '>>',\
    os.environ['TMP_DIR_WIN'] + '/ci_scripts/pytorch_env_restore_helper.bat'])

    subprocess.run(['aws', 's3', 'cp', '\"s3://ossci-windows/Restore PyTorch Environment.lnk\"',\
    '\"C:\\Users\\circleci\\Desktop\\Restore PyTorch Environment.lnk\"'])


subprocess.call("python setup.py bdist_wheel", shell=True)
subprocess.call("sccache --show-stats", shell=True)
subprocess.call("python -c \"import os, glob; os.system(\'python -mpip install \' + glob.glob(\'dist/*.whl\')[0] + \'[opt-einsum]\')\"", shell=True)


if 'BUILD_ENVIRONMENT' not in os.environ:
    subprocess.call('echo NOTE: To run \`import torch\`, please make sure to activate the conda environment by running \`call ' +\
    os.environ['CONDA_PARENT_DIR'] + '\\Miniconda3\\Scripts\\activate.bat ' + os.environ['CONDA_PARENT_DIR'] + '\\Miniconda3\'' +\
    ' in Command Prompt before running Git Bash.', shell=True)

else:
    subprocess.call('7z a ' + os.environ['TMP_DIR_WIN'] + '\\' + os.environ['IMAGE_COMMIT_TAG'] + '.7z ' +\
    os.environ['CONDA_PARENT_DIR'] + '\\Miniconda3\\Lib\\site-packages\\torch ' + os.environ['CONDA_PARENT_DIR'] +\
    '\\Miniconda3\\Lib\\site-packages\\torchgen ' + os.environ['CONDA_PARENT_DIR'] +\
    '\\Miniconda3\\Lib\\site-packages\\functorch \&\& copy /Y \"' + os.environ['TMP_DIR_WIN'] +\
    '\\' + os.environ['IMAGE_COMMIT_TAG'] + '.7z\" \"' + os.environ['PYTORCH_FINAL_PACKAGE_DIR'] + '\\\"', shell=True)

    # export test times so that potential sharded tests that'll branch off this build will use consistent data
    subprocess.call('python tools/stats/export_test_times.py', shell=True)
    shutil.copy(".pytorch-test-times.json", os.environ['PYTORCH_FINAL_PACKAGE_DIR'])

    # Also save build/.ninja_log as an artifact
    shutil.copy("build\\.ninja_log", os.environ['PYTORCH_FINAL_PACKAGE_DIR'] + '\\')




subprocess.call('sccache --show-stats --stats-format json | jq .stats > sccache-stats-' +\
os.environ['BUILD_ENVIRONMENT'] + '-' + os.environ['OUR_GITHUB_JOB_ID'] + '.json', shell=True)

subprocess.call('sccache --stop-server', shell=True)
