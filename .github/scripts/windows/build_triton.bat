@echo on

set DESIRED_PYTHON=%PY_VERS%
call .ci/pytorch/windows/internal/install_python.bat

:: Fix cmake version for issue https://github.com/pytorch/pytorch/issues/150480
:: Cython 3.3.0's compiled cp315 extension access-violates (0xC0000005) under
:: CPython 3.15, crashing `setup.py bdist_wheel`. Same defect as gh-194618.
%PYTHON_EXEC% -m pip install wheel pybind11 certifi "cython<3.3.0" cmake==3.31.6 setuptools==72.1.0 ninja==1.11.1.4

dir "%VC_INSTALL_PATH%"

call "%VC_INSTALL_PATH%\VC\Auxiliary\Build\vcvarsall.bat" x64
%PYTHON_EXEC% .github/scripts/build_triton_wheel.py --device=%BUILD_DEVICE% %RELEASE%
