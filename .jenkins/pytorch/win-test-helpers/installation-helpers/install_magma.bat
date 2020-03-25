if "%CUDA_VERSION%" == "9" set CUDA_SUFFIX=cuda92
if "%CUDA_VERSION%" == "10" set CUDA_SUFFIX=cuda101

if "%CUDA_SUFFIX%" == "" (
  echo unknown CUDA version, please set `CUDA_VERSION` to 9 or 10.
  exit /b 1
)

if "%REBUILD%"=="" (
  if "%BUILD_ENVIRONMENT%"=="" (
    curl --retry 3 -k https://s3.amazonaws.com/ossci-windows/magma_2.5.2_%CUDA_SUFFIX%_%BUILD_TYPE%.7z --output %TMP_DIR_WIN%\magma_2.5.2_%CUDA_SUFFIX%_%BUILD_TYPE%.7z
  ) else (
    aws s3 cp s3://ossci-windows/magma_2.5.2_%CUDA_SUFFIX%_%BUILD_TYPE%.7z %TMP_DIR_WIN%\magma_2.5.2_%CUDA_SUFFIX%_%BUILD_TYPE%.7z --quiet
  )
  7z x -aoa %TMP_DIR_WIN%\magma_2.5.2_%CUDA_SUFFIX%_%BUILD_TYPE%.7z -o%TMP_DIR_WIN%\magma
)
set MAGMA_HOME=%TMP_DIR_WIN%\magma
