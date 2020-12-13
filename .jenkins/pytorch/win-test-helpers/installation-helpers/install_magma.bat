rem remove dot in cuda_version, fox example 11.1 to 111
set VERSION_SUFFIX=%CUDA_VERSION:.=%
set CUDA_SUFFIX=cuda%VERSION_SUFFIX%

if "%CUDA_SUFFIX%" == "" (
  echo unknown CUDA version, please set `CUDA_VERSION` higher than 9.2
  exit /b 1
)

if "%REBUILD%"=="" (
  if "%BUILD_ENVIRONMENT%"=="" (
    curl --retry 3 -k https://s3.amazonaws.com/ossci-windows/magma_2.5.4_%CUDA_SUFFIX%_%BUILD_TYPE%.7z --output %TMP_DIR_WIN%\magma_2.5.4_%CUDA_SUFFIX%_%BUILD_TYPE%.7z
  ) else (
    aws s3 cp s3://ossci-windows/magma_2.5.4_%CUDA_SUFFIX%_%BUILD_TYPE%.7z %TMP_DIR_WIN%\magma_2.5.4_%CUDA_SUFFIX%_%BUILD_TYPE%.7z --quiet
  )
  7z x -aoa %TMP_DIR_WIN%\magma_2.5.4_%CUDA_SUFFIX%_%BUILD_TYPE%.7z -o%TMP_DIR_WIN%\magma
)
set MAGMA_HOME=%TMP_DIR_WIN%\magma
