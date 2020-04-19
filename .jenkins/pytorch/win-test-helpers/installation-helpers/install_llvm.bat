mkdir %TMP_DIR_WIN%\bin

if "%REBUILD%"=="" (
  if "%BUILD_ENVIRONMENT%"=="" (
    curl --retry 3 -k https://s3.amazonaws.com/ossci-windows/LLVM-10.0.0-win64.exe --output %TMP_DIR_WIN%\LLVM-10.0.0-win64.exe
  ) else (
    aws s3 cp s3://ossci-windows/LLVM-10.0.0-win64.exe %TMP_DIR_WIN%\LLVM-10.0.0-win64.exe --quiet
  )
  7z x -aoa %TMP_DIR_WIN%\LLVM-10.0.0-win64.exe -o"C:\Program Files\LLVM"
)
