@echo on

set DESIRED_PYTHON=%PY_VERS%
call .ci/pytorch/windows/internal/install_python.bat

:: Fix cmake version for issue https://github.com/pytorch/pytorch/issues/150480
%PYTHON_EXEC% -m pip install wheel pybind11 certifi cython cmake==3.31.6 setuptools==72.1.0 ninja==1.11.1.4

dir "%VC_INSTALL_PATH%"

call "%VC_INSTALL_PATH%\VC\Auxiliary\Build\vcvarsall.bat" x64
%PYTHON_EXEC% .github/scripts/build_triton_wheel.py --device=%BUILD_DEVICE% %RELEASE%
