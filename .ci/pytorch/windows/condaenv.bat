IF "%DESIRED_PYTHON%"=="" (
    echo DESIRED_PYTHON is NOT defined.
    exit /b 1
)

:: Create a new conda environment
setlocal EnableDelayedExpansion
FOR %%v IN (%DESIRED_PYTHON%) DO (
    set PYTHON_VERSION_STR=%%v
    set NUMPY_VERSION=2.0.1
    set PYTHON_VERSION_STR=!PYTHON_VERSION_STR:.=!
    conda remove -n py!PYTHON_VERSION_STR! --all -y || rmdir %CONDA_HOME%\envs\py!PYTHON_VERSION_STR! /s
    if "%%v" == "3.13t" (
        call conda create -n py!PYTHON_VERSION_STR! -y -c=conda-forge python-freethreading python=3.13
        set NUMPY_VERSION=2.1.2
    ) else (
        call conda create -n py!PYTHON_VERSION_STR! -y python=%%v
    )
    if "%%v" == "3.13" set NUMPY_VERSION=2.1.2
    call conda run -n py!PYTHON_VERSION_STR! pip install pyyaml boto3 cmake ninja typing_extensions setuptools==72.1.0 numpy==!NUMPY_VERSION!
    call conda run -n py!PYTHON_VERSION_STR! pip install mkl-static mkl-include
)
endlocal

:: Install libuv
conda install -y -q -c conda-forge libuv=1.39
set libuv_ROOT=%CONDA_HOME%\Library
echo libuv_ROOT=%libuv_ROOT%
