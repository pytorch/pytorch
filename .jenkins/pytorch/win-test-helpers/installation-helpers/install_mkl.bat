if "%REBUILD%"=="" (
  if "%BUILD_ENVIRONMENT%"=="" (
    curl -k https://s3.amazonaws.com/ossci-windows/mkl_2019.4.245.7z --output %TMP_DIR_WIN%\mkl.7z
  ) else (
    aws s3 cp s3://ossci-windows/mkl_2019.4.245.7z %TMP_DIR_WIN%\mkl.7z --quiet
  )
  7z x -aoa %TMP_DIR_WIN%\mkl.7z -o%TMP_DIR_WIN%\mkl
)
set CMAKE_INCLUDE_PATH=%TMP_DIR_WIN%\mkl\include
set LIB=%TMP_DIR_WIN%\mkl\lib;%LIB
