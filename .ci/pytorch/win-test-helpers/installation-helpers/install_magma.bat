if "%CUDA_VERSION%" == "cpu" (
  echo skip magma installation for cpu builds
  exit /b 0
)

rem remove dot in cuda_version, fox example 11.1 to 111

if not "%USE_CUDA%"=="1" (
    exit /b 0
)

if x%CUDA_VERSION:.=%==x%CUDA_VERSION% (
    echo CUDA version %CUDA_VERSION% format isn't correct, which doesn't contain '.'
    exit /b 1
)

set VERSION_SUFFIX=%CUDA_VERSION:.=%
set CUDA_SUFFIX=cuda%VERSION_SUFFIX%

if "%CUDA_SUFFIX%" == "" (
  echo unknown CUDA version, please set `CUDA_VERSION` higher than 10.2
  exit /b 1
)

if "%REBUILD%"=="" (
  if "%BUILD_ENVIRONMENT%"=="" (
    curl --retry 3 --retry-all-errors -k https://s3.amazonaws.com/ossci-windows/magma_2.5.4_%CUDA_SUFFIX%_%BUILD_TYPE%.7z --output %TMP_DIR_WIN%\magma_2.5.4_%CUDA_SUFFIX%_%BUILD_TYPE%.7z
  ) else (
    aws s3 cp s3://ossci-windows/magma_2.5.4_%CUDA_SUFFIX%_%BUILD_TYPE%.7z %TMP_DIR_WIN%\magma_2.5.4_%CUDA_SUFFIX%_%BUILD_TYPE%.7z --quiet
  )
  if errorlevel 1 exit /b
  if not errorlevel 0 exit /b
  7z x -aoa %TMP_DIR_WIN%\magma_2.5.4_%CUDA_SUFFIX%_%BUILD_TYPE%.7z -o%TMP_DIR_WIN%\magma
  if errorlevel 1 exit /b
  if not errorlevel 0 exit /b
)
set MAGMA_HOME=%TMP_DIR_WIN%\magma
