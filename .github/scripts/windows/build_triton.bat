@echo on

set PYTHON_PREFIX=%PY_VERS:.=%
set PYTHON_PREFIX=py%PYTHON_PREFIX:;=;py%
call .ci/pytorch/win-test-helpers/installation-helpers/activate_miniconda3.bat
:: Create a new conda environment
if "%PY_VERS%" == "3.13t" (
    call conda create -n %PYTHON_PREFIX% -y -c=conda-forge python-freethreading python=3.13
) else (
    call conda create -n %PYTHON_PREFIX% -y -c=conda-forge python=%PY_VERS%
)
:: Fix cmake version for issue https://github.com/pytorch/pytorch/issues/150480
call conda run -n %PYTHON_PREFIX% pip install wheel pybind11 certifi cython cmake==3.31.6 setuptools==72.1.0 ninja

dir "%VC_INSTALL_PATH%"

call "%VC_INSTALL_PATH%\VC\Auxiliary\Build\vcvarsall.bat" x64
call conda run -n %PYTHON_PREFIX% python .github/scripts/build_triton_wheel.py --device=%BUILD_DEVICE% %RELEASE%
