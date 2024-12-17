if "%DEBUG%" == "1" (
  set BUILD_TYPE=debug
) ELSE (
  set BUILD_TYPE=release
)

:: This inflates our log size slightly, but it is REALLY useful to be
:: able to see what our cl.exe commands are (since you can actually
:: just copy-paste them into a local Windows setup to just rebuild a
:: single file.)
:: log sizes are too long, but leaving this here incase someone wants to use it locally
:: set CMAKE_VERBOSE_MAKEFILE=1


set INSTALLER_DIR=%SCRIPT_HELPERS_DIR%\installation-helpers

@echo on
popd

:: The default sccache idle timeout is 600, which is too short and leads to intermittent build errors.
set SCCACHE_IDLE_TIMEOUT=0
set SCCACHE_IGNORE_SERVER_IO_ERROR=1
sccache --stop-server
sccache --start-server
sccache --zero-stats
set CMAKE_C_COMPILER_LAUNCHER=sccache
set CMAKE_CXX_COMPILER_LAUNCHER=sccache

:: environment variables
set BLAS=APL
set USE_LAPACK=1
set CMAKE_BUILD_TYPE=%BUILD_TYPE%
if defined PYTORCH_BUILD_VERSION (
  set PYTORCH_BUILD_VERSION=%PYTORCH_BUILD_VERSION%
  set PYTORCH_BUILD_NUMBER=1
)

:: Print all existing environment variable for debugging
set

:: activate visual studio
call "%DEPENDENCIES_DIR%\VSBuildTools\VC\Auxiliary\Build\vcvarsall.bat" arm64 -vcvars_ver=%MSVC_VERSION%
where cl.exe

:: create virtual environment
python -m venv .venv
echo * > .venv\.gitignore
call .\.venv\Scripts\activate
where python

python setup.py bdist_wheel
if errorlevel 1 goto fail
if not errorlevel 0 goto fail
sccache --show-stats
python -c "import os, glob; os.system('python -mpip install --no-index --no-deps ' + glob.glob('dist/*.whl')[0])"
(
  if "%BUILD_ENVIRONMENT%"=="" (
    echo NOTE: To run `import torch`, please make sure to activate the conda environment by running `call %CONDA_PARENT_DIR%\Miniconda3\Scripts\activate.bat %CONDA_PARENT_DIR%\Miniconda3` in Command Prompt before running Git Bash.
  ) else (
    copy /Y "dist\*.whl" "%PYTORCH_FINAL_PACKAGE_DIR%"

    :: export test times so that potential sharded tests that'll branch off this build will use consistent data
    python tools/stats/export_test_times.py
    robocopy /E ".additional_ci_files" "%PYTORCH_FINAL_PACKAGE_DIR%\.additional_ci_files"

  )
)

sccache --show-stats --stats-format json | jq .stats > sccache-stats-%BUILD_ENVIRONMENT%-%OUR_GITHUB_JOB_ID%.json
sccache --stop-server

exit /b 0

:fail
exit /b 1
