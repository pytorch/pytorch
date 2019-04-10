if "%REBUILD%"=="" (
  if "%BUILD_ENVIRONMENT%"=="" (
    curl -k https://s3.amazonaws.com/ossci-windows/magma_2.5.0_cuda90_%BUILD_TYPE%.7z --output %TMP_DIR_WIN%\magma_2.5.0_cuda90_%BUILD_TYPE%.7z
  ) else (
    aws s3 cp s3://ossci-windows/magma_2.5.0_cuda90_%BUILD_TYPE%.7z %TMP_DIR_WIN%\magma_2.5.0_cuda90_%BUILD_TYPE%.7z --quiet
  )
  7z x -aoa %TMP_DIR_WIN%\magma_2.5.0_cuda90_%BUILD_TYPE%.7z -o%TMP_DIR_WIN%\magma
)
set MAGMA_HOME=%TMP_DIR_WIN%\magma
